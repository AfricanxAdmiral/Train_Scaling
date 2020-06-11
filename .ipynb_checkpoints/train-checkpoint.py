import copy
import random
import time

import numpy as np
import torch
import csv

#from inclearn.lib import factory, results_utils, utils
from lib import factory, results_utils, utils

from sklearn.metrics import confusion_matrix
import seaborn as sn
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    order_list = copy.deepcopy(args["order"])
    for seed in seed_list:
        for order in order_list:
            
            with open('../results/results_txt/cifar_learn50_(1000)_ours_without_Norm_classifier.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")
            with open('../results/results_txt/cifar_learn50_(1000)_ours_without_Norm_nearest.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")
            """ 
            with open('../results/results_txt/cifar_(2000)_ours_04dissEweightDist_nobias_classifier_top5.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")
            with open('../results/results_txt/cifar_(2000)_ours_04dissEweightDist_nobias_nearest_top5.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")
            
            with open('../results/results_json/cifar_(2000)_ours_04dissEweightDist_nobias.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")
            """ 
            args["seed"] = seed
            args["device"] = device
            args["order"] = order
            start_time = time.time()
            _train(args)
            print("Training finished in {}s.".format(int(time.time() - start_time)))


def _train(args):
    _set_seed(args["seed"])

    factory.set_device(args)

    inc_dataset = factory.get_data(args)
    args["classes_order"] = inc_dataset.class_order
    print(inc_dataset.class_order)
    model = factory.get_model(args)

    results = results_utils.get_template_results(args)

    memory = None

    for _ in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory)
        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=task_info["max_task"]
        )

        model.eval()
        model.before_task(train_loader, val_loader)
        print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train()
        model.train_task(train_loader, val_loader)
        model.eval()
        model.after_task(inc_dataset)
        #model._save_model("../model_weight/cifar_(2000)_ours_005009dissEweightDist_v2_withoutbias_order{}_task{}.pkl".format(args["order"], task_info["task"]))

        print("Eval on {}->{}.".format(0, task_info["max_class"]))
        #ypred, ytrue = model.eval_task(test_loader)
        ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(test_loader)
        
        acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
        print(acc_stats)
        results["results"].append(acc_stats)
        
        ####################################################################################
        #classifier 100 classes
        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
        print('classifier:     ',acc_stats)
        
        with open('../results/results_txt/cifar_(1000)_learn50_ours_without_Norm_classifier.txt', "a") as accuracy:
            for i in acc_stats.values():
                accuracy.write(str(i) + " ")
            accuracy.write("\n")
        """     
        #top5
        acc_stats = utils.compute_accuracy(y_top5, ytrue, task_size=args["increment"])
        print('classifier top5:',acc_stats)
        
        with open('../results/results_txt/cifar_(2000)_ours_04dissEweightDist_nobias_classifier_top5.txt', "a") as accuracy:
            for i in acc_stats.values():
                accuracy.write(str(i) + " ")
            accuracy.write("\n")
        """     
        #nearest 100 classes
        acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
        print('nearest:        ',acc_stats)
        
        with open('../results/results_txt/cifar_(1000)_learn50_ours_without_Norm_nearest.txt', "a") as accuracy:
            for i in acc_stats.values():
                accuracy.write(str(i) + " ")
            accuracy.write("\n")
        """     
        #top5
        acc_stats = utils.compute_accuracy(yn_top5, yntrue, task_size=args["increment"])
        print('nearest top5:   ',acc_stats)

        with open('../results/results_txt/cifar_(2000)_ours_04dissEweightDist_nobias_nearest_top5.txt', "a") as accuracy:
            for i in acc_stats.values():
                accuracy.write(str(i) + " ")
            accuracy.write("\n")
        """  
        """ 
        ###############################Confusion matrix######################################
        if _ == (inc_dataset.n_tasks-1) :
            confusion = confusion_matrix(ytrue, ypred)
            confusion_plot = sn.heatmap(confusion, annot=False, cbar=False,
                     xticklabels =10,yticklabels =10, square = True)
            fig = confusion_plot.get_figure()
            fig.savefig("../confusion_matrix/shuffle/cifar_(2000)_ours_04dissEweightDist_nobias/classifier"+str(args["order"])+".png")
            
            confusion = confusion_matrix(yntrue, ynpred)
            confusion_plot = sn.heatmap(confusion, annot=False, cbar=False,
                     xticklabels =10,yticklabels =10, square = True)
            fig = confusion_plot.get_figure()
            fig.savefig("../confusion_matrix/shuffle/cifar_(2000)_ours_04dissEweightDist_nobias/nearest"+str(args["order"])+".png")
        #####################################################################################
        """ 
        memory = model.get_memory()

    print(
        "Average Incremental Accuracy: {}.".format(
            results_utils.compute_avg_inc_acc(results["results"])
        )
    )

    if args["name"]:
        results_utils.save_results(results, args["name"])

    del model
    del inc_dataset
    torch.cuda.empty_cache()


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
