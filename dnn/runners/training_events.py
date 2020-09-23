from ignite.handlers import Timer
from ignite.engine import Events
import ignite
import matplotlib.pyplot as plt
import numpy as np
from dnn.typing_definitions import *
import pandas as pd

from visualize_utils.make_metric_and_xray_figure import xray_figure, create_metric_table, plot_precision_recall, plot_roc
from visualize_utils.net_results_vis import tsne, knn_analysis
from visualize_utils.save_graphs import save_graphs
import os
import os.path as osp
import seaborn as sns


def set_training_epoch(engine, starting_epoch):
    engine.state.epoch = starting_epoch
    data_size = len(engine.state.dataloader)  # num of batches
    engine.state.iteration = data_size * engine.state.epoch


def log_iter_complete_screen(engine, timer, to_print=True):
    if to_print:
        data_size = len(engine.state.dataloader)  # number of batches
        time_per_image = timer.value() / engine.state.dataloader.batch_sampler.batch_size
        print(
            f"Training Epoch [{engine.state.epoch}]"
            f" | Iter [{engine.state.iteration - data_size * (engine.state.epoch - 1)}/{data_size}]"
            f" | Loss: {engine.state.output}"
            f" | Time per image: {round(time_per_image, 4)}"
        )


def log_iter_complete_tb(engine, summary_writer):
    loss = engine.state.output
    summary_writer.add_scalar('loss', loss, engine.state.iteration)


def log_epoch_start_screen(engine, to_print=True):
    if to_print:
        print(f"Starting Epoch {engine.state.epoch}")


def log_epoch_end_screen(engine, timer, to_print=True):
    if to_print:
        print(f"Epoch total time: {round(timer.value(), 4)}")


def log_train_end_screen(engine, summary_writer, timer, to_print=True):
    if to_print:
        print(f"Training total time: {round(timer.value(), 4)}")

    summary_writer.close()


def log_eval_iter_screen(engine, timer, to_print=True):
    if to_print:
        data_size = len(engine.state.dataloader)  # number of batches
        time_per_image = timer.value() / engine.state.dataloader.batch_sampler.batch_size
        print(f"Eval Iter [{engine.state.iteration - data_size * (engine.state.epoch - 1)}/{data_size}]"
              f" | Time per image: {round(time_per_image, 4)}")


def save_checkpoint(engine, checkpoint_handler, model, optimizer, trainer, lr_scheduler, num_attempts=5):
    for _ in range(num_attempts):
        try:
            checkpoint_handler(engine, {"trainer": trainer,
                                        "model": model,
                                        "optimizer": optimizer,
                                        })
            lr_scheduler.step()
            return
        except OSError:
            print(f"Failed at saving checkpoint {engine.state.epoch}")

def image_and_table_figure_maker(df, dir_path):
    dir_path = osp.join(dir_path, 'figures')
    os.makedirs(dir_path, exist_ok=True)
    #xray_figure(df, dir_path)
    create_metric_table(df, dir_path)
    plot_roc(df, dir_path)
    plot_precision_recall(df, dir_path)

def save_results_to_csv(dir_path, metric_results):
    def plot_confidence(df):
       
        df_covid = df.loc[df['gt'] == 1, ['corona_confidence']].values
        df_non_covid = df.loc[df['gt'] == 0, ['corona_confidence']].values

        plt.hist(df_covid, bins = 30)
        plt.hist(df_non_covid, alpha = 0.7, bins = 30)
        plt.ylabel("frequency")
        plt.xlabel("confidence of image labeling")
        plt.legend(["GT: positive for Covid-19", "GT: negative for Covid-19"])
        plt.axvline(x=0.5,color='gray', linestyle='--')
        plot_path = osp.join(dir_path, 'confidence_plot.png')
        plt.savefig(plot_path)

    df_dict = {}
    y_pred, y, conf, img_name = metric_results
    df_dict['image_name'] = img_name
    df_dict['gt'] = y
    df_dict['net_prediction'] = y_pred
    df_dict['corona_confidence'] = conf[:, 1]
    df_dict['no_corona_confidence'] = conf[:, 0]
    df_dict['mistake'] = ~(y==y_pred)

    df = pd.DataFrame(df_dict)
    csv_path = osp.join(dir_path, 'results.csv')
    df.to_csv(csv_path)

    plot_confidence(df)
    # image_and_table_figure_maker(df, dir_path)



def print_metrics_after_eval(metrics, epoch, timer, mode):
    print(f"Data set: {mode}"
          f" | End of Epoch: {epoch}"
          f" | Recall: {metrics['recall']}"
          f" | Precision: {metrics['precision']}"
          f" | Accuracy: {metrics['accuracy']}"
          f" | Eval total time: {round(timer.value(), 4)}")


def create_confusion_matrix_fig(cm):
    fig, ax = plt.subplots()
    cm_normed = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    cm_normed = pd.DataFrame(cm_normed, index=['not_corona', 'corona'], columns=['not_corona', 'corona']).values*100
    cm = pd.DataFrame(cm, index=['not_corona', 'corona'], columns=['not_corona', 'corona']).values

    label = np.array([f"{normed_val:.2f}% \n ({val})" for val, normed_val in zip(cm.flatten(), cm_normed.flatten())])\
        .reshape(cm.shape[0], cm.shape[1])
    sns.heatmap(cm_normed, annot=label, annot_kws={'size': 14}, fmt='')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    return fig


def evaluate_model_during_training(engine, evaluator, data_loader_dict, timer, eval_freq, summary_writer):
    if engine.state.epoch % eval_freq != 0:
        return

    for mode, data_loader in data_loader_dict.items():
        evaluator.run(data_loader)
        metrics = evaluator.state.metrics
        print_metrics_after_eval(metrics=metrics, epoch=engine.state.epoch, timer=timer, mode=mode)

        summary_writer.add_scalars('accuracy', {f'{mode}': metrics['accuracy']}, engine.state.epoch)
        summary_writer.add_scalars('precision', {f'{mode}': metrics['precision']}, engine.state.epoch)
        summary_writer.add_scalars('recall', {f'{mode}': metrics['recall']}, engine.state.epoch)

        confusion_matrix = metrics['confusion_matrix'].numpy()
        confusion_matrix_fig = create_confusion_matrix_fig(cm=confusion_matrix)
        summary_writer.add_figure(f'confusion matrix_{mode}', confusion_matrix_fig, engine.state.epoch)


def evaluate_model_end_of_training(engine, evaluator, data_loader_dict, logging_dir, timer):
    for name, data_loader in data_loader_dict.items():
        evaluator.run(data_loader)
        evaluate_model_without_training(evaluator, timer=timer, eval_dir=logging_dir, data_set=name)


def evaluate_model_without_training(engine, timer, eval_dir, data_set: str):
    metrics = engine.state.metrics
    print(f"Data set: {data_set}"
          f" | Recall: {metrics['recall']}"
          f" | Precision: {metrics['precision']}"
          f" | Accuracy: {metrics['accuracy']}"
          f" | Eval total time: {round(timer.value(), 4)}")

    confusion_matrix = metrics['confusion_matrix'].numpy()
    confusion_matrix_fig = create_confusion_matrix_fig(cm=confusion_matrix)
    save_graphs(dir_path=eval_dir, graph=confusion_matrix_fig, name=data_set)

    metric_output_results = metrics['metric_output_results']
    metric_last_layer = metrics['metric_last_layer']
    save_results_to_csv(dir_path=eval_dir, metric_results=metric_output_results)
    knn_analysis(feature_map=metric_last_layer, results=metric_output_results, dir_path=eval_dir)
    tsne(feature_map=metric_last_layer, results=metric_output_results, component_num=2, dir_path=eval_dir)
    save_embeddings(dir_path=eval_dir, feature_map=metric_last_layer, results=metric_output_results)

def save_embeddings(dir_path, feature_map, results):
    import pickle
    y_pred, y, conf, img_name = results
    filename = osp.join(dir_path, "img_embeddings.pkl")
    embeddings_dict = {"embeddings": feature_map,
                       "label": y,
                       "img_names" : img_name
                       }
    with open(filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)

def attach_trainer_events(trainer, evaluator, train_data_loader, test_data_loader, checkpoint_handler, model,
                          summary_writer, eval_freq, optimizer, eval_dir, lr_scheduler, starting_epoch=0):

    # Timers initializations
    timer_iter_train = Timer()
    timer_iter_train.attach(trainer, start=Events.ITERATION_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    timer_iter_eval = Timer()
    timer_iter_eval.attach(evaluator, start=Events.ITERATION_STARTED, resume=Events.ITERATION_STARTED,
                           pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    timer_epoch_train = Timer()
    timer_epoch_train.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.EPOCH_STARTED,
                             pause=Events.EPOCH_COMPLETED, step=Events.EPOCH_COMPLETED)

    timer_epoch_eval = Timer()
    timer_epoch_eval.attach(evaluator, start=Events.EPOCH_STARTED, resume=Events.EPOCH_STARTED,
                            pause=Events.EPOCH_COMPLETED, step=Events.EPOCH_COMPLETED)

    timer_train = Timer()
    timer_train.attach(trainer, start=Events.STARTED, pause=Events.COMPLETED)

    # Starting trainer events
    trainer.add_event_handler(Events.STARTED, set_training_epoch, starting_epoch=starting_epoch)

    # Iteration complete event
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_iter_complete_screen, timer=timer_iter_train)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_iter_complete_tb, summary_writer=summary_writer)

    # Epoch started event
    trainer.add_event_handler(Events.EPOCH_STARTED, log_epoch_start_screen)

    # Epoch ended events
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_checkpoint, checkpoint_handler=checkpoint_handler,
                              model=model, optimizer=optimizer, trainer=trainer, lr_scheduler=lr_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_epoch_end_screen, timer=timer_epoch_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_during_training, evaluator=evaluator,
                              data_loader_dict={'train': train_data_loader, 'test': test_data_loader},
                              timer=timer_epoch_eval, eval_freq=eval_freq, summary_writer=summary_writer)

    # Ending trainer events
    # trainer.add_event_handler(Events.COMPLETED, evaluate_model_end_of_training, evaluator=evaluator,
    #                           data_loader_dict={'test': test_data_loader},
    #                           timer=timer_epoch_eval, logging_dir=eval_dir)

    trainer.add_event_handler(Events.COMPLETED, log_train_end_screen, summary_writer=summary_writer, timer=timer_train)

    # Evaluator iteration events
    # evaluator.add_event_handler(Events.ITERATION_COMPLETED, log_eval_iter_screen, timer=timer_iter_eval)

def attach_evaluator_events(evaluator, experiment_dir, data_set: str):

    # Timers initializations
    timer_iter = Timer()
    timer_iter.attach(evaluator, start=Events.ITERATION_STARTED, resume=Events.ITERATION_STARTED,
                      pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    timer_epoch = Timer()
    timer_epoch.attach(evaluator, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Evaluator iteration events
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, log_eval_iter_screen, timer=timer_iter)

    # Ending of evaluation events
    eval_dir = osp.join(experiment_dir, 'inference_results')
    evaluator.add_event_handler(Events.COMPLETED, evaluate_model_without_training, eval_dir=eval_dir, timer=timer_epoch,
                                data_set=data_set)
