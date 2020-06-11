import copy
import random
import time

import numpy as np
import torch
import csv

#from inclearn.lib import factory, results_utils, utils
from lib import factory, results_utils, utils

from sklearn.metrics import confusion_matrix
from convnet import conv2d_fw

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    order_list = copy.deepcopy(args["order"])
    for seed in seed_list:
        for order in order_list:
            
            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")
            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as annotation:
                annotation.write("###########This is order"+str(order)+"####################\n")

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
    
    # build ltl data loader (train:val = 0.5:0.5)
    ltl_inc_dataset = factory.get_ltl_data(args)
    # build ss data loader (batch size=1)
    ss_inc_dataset = factory.get_ss_data(args)
    gb_inc_dataset = factory.get_gb_data(args)
    
    args["classes_order"] = inc_dataset.class_order
    print(inc_dataset.class_order)
    model = factory.get_model(args)
    
    results = results_utils.get_template_results(args)

    memory = None
    
    ###load pretrain model (metric)######################
    #model._network.load_state_dict(torch.load('./net.weight'))
    model._metric_network.load_state_dict(torch.load('../../iCaRL_new/model_weight/cifar_pretrain_metric_withNorm_nearest.pkl'))
    #freeze
    for param in model._metric_network.parameters():
        param.requires_grad = False
    model._metric_network.eval()
    ######################################
    
    for _ in range(inc_dataset.n_tasks):
        # the data loader is for pretraining model 
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory)
        
        # the data loader is for ss generator(nt yet implement)
        task_info, ss_train_loader, ss_val_loader, ss_test_loader = ss_inc_dataset.new_task(memory)        

        
        if task_info["task"] == args["max_task"]:
            break
        
        # prepare to learn to learn (1st pretrain, 2nd ltl)
        print("#################"+str(task_info["task"])+"#####################################")
        if task_info["task"] == 0:
            # the data loader is for training the gamma and beta (in cross domain method)
            ltl_task_info, ltl_train_loader, ltl_val_loader, ltl_test_loader = ltl_inc_dataset.new_task(memory)
            
            # get the data loader for train 100 classes
            # the data loader is for training 100 classes together (to see the scaling and shifting can work or not)
            gb_task_info, gb_train_loader, gb_val_loader, gb_test_loader = gb_inc_dataset.new_task(None)
            
            # 1st pretrain
            model.set_task_info(
                task=task_info["task"],
                total_n_classes=task_info["max_class"],
                increment=task_info["increment"],
                n_train_data=task_info["n_train_data"],
                n_test_data=task_info["n_test_data"],
                n_tasks=task_info["max_task"]
            )
            # adjust the controller of mtl weight, feature wise layer and need_grad
            #change_ft(model, ft=False)
            #change_mtl(model, mtl=False)
            #change_ss_flag(model, flag=False) 
            #change_weight_requires_grad(model, normal_grad_need=True, mtl_grad_need=False)
            #change_fw_requires_grad(model, fw_need=False)
            
            model.eval()
            model.before_task(train_loader, val_loader)
            print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            """
            model.train()
            model.train_task(train_loader, val_loader)
            """
            model._network.load_state_dict(torch.load('./pre_net.weight'))
            model.eval()
            # torch.save(model._network.state_dict(), './pre_net.weight')
            model.after_task(inc_dataset)
            
            print("Eval on {}->{}.".format(0, task_info["max_class"]))
            
            # classify 100 class
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(train_loader)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('train classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('train nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            ####################################################################################
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(test_loader)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            ####################################################################################
            
            # 2nd ltl
            model.set_task_info(
                task=ltl_task_info["task"],
                total_n_classes=ltl_task_info["max_class"],
                increment=ltl_task_info["increment"],
                n_train_data=ltl_task_info["n_train_data"],
                n_test_data=ltl_task_info["n_test_data"],
                n_tasks=ltl_task_info["max_task"]
            )
            # use feature wise during training
            #change_ft(model, ft=True)

            model.eval()
            model.before_task_to_2nd_ltl(ltl_train_loader, ltl_val_loader)
            print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            
            # model.train()
            # model.ltl_train_task(ltl_train_loader, ltl_val_loader)
            
            model._network.load_state_dict(torch.load('./ss_net.weight'))
            model.eval()
            # torch.save(model._network.state_dict(), './ss_net.weight')
            model.after_task(inc_dataset)
            
            print("Eval on {}->{}.".format(0, task_info["max_class"]))
            
            # classify 100 class
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(train_loader)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('train classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('train nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            ####################################################################################
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(test_loader)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            ####################################################################################
            
            # train the ss generator
            # close the ft and open the mtl to train the scaling and shifting
            #change_ft(model, ft=False)
            #change_mtl(model, mtl=True)
            #change_ss_flag(model, flag=True)
            
            
            # fixed the normal weight 
            #chage_extractor_requires_grad(model, ex_need_1=False, ex_need_2=False, ex_need_3=False, ex_need=False)            

            # for the scaling and shifting (gamma beta)
            #chage_extractor_requires_grad_mtl(model, ex_need_1=False, ex_need_2=False, ex_need_3=False, ex_need_1_bn_fw=False, ex_need_2_bn_fw=False, ex_need_3_bn_fw=False, ex_need=False) # <- not need now
            
            #chage_extractor_requires_grad_gb(model, ex_need_1_bn_fw=True, ex_need_2_bn_fw=True, ex_need_3_bn_fw=True, ex_need=True)
            
            # train the scaling and shifting
            # gb_train_loader: 100 classes 
            model.before_ss(train_loader, val_loader, ss=True)
            model.train()
            model.train_ss(train_loader, val_loader)
            model.eval()
            
            # use inc_dataset to store the exemplars
#             model.after_task(inc_dataset)
#             model.after_task(ss_inc_dataset)
#             model.after_task(inc_dataset)
#             model.ss_after_task(ss_inc_dataset)

            print('finish!!!!!!!!')
            # gb_after_task: dosen't store the exemplars
            model.gb_after_task(inc_dataset)
            
            print("Eval on {}->{}.".format(0, task_info["max_class"]))
            
            # classify 100 class
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(train_loader, ss=1)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('train classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('train nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            ####################################################################################
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(test_loader, ss=1)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            memory = model.get_memory()
            ####################################################################################
            
        #elif task_info["task"] == 1:    
        #    break
            
        else:  # below not yet modified, need to modify when use ss generator          
            #change_ft(model, ft=False)
            #change_mtl(model, mtl=True)
            #change_ss_flag(model, flag=True)
                        
            # close the updates
            #chage_extractor_requires_grad(model, ex_need_1=False, ex_need_2=False, ex_need_3=False, ex_need=False)
            
            # for meta-transfer learning        
            #chage_extractor_requires_grad_mtl(model, ex_need_1=True, ex_need_2=True, ex_need_3=True, ex_need_1_bn_fw=True, ex_need_2_bn_fw=True, ex_need_3_bn_fw=True, ex_need=True)    
    
            # only open the mtl_weight of the stage_1, stage_2 and stage_3
#             chage_extractor_requires_grad_mtl(model, ex_need_1=True, ex_need_2=True, ex_need_3=True, ex_need_1_bn_fw=False, ex_need_2_bn_fw=False, ex_need_3_bn_fw=False)

             # only open the mtl_weight of the stage_3
#             chage_extractor_requires_grad_mtl(model, ex_need_1=False, ex_need_2=False, ex_need_3=True, ex_need_1_bn_fw=False, ex_need_2_bn_fw=False, ex_need_3_bn_fw=False)

            
            model.set_task_info(
                task=task_info["task"],
                total_n_classes=task_info["max_class"],
                increment=task_info["increment"],
                n_train_data=task_info["n_train_data"],
                n_test_data=task_info["n_test_data"],
                n_tasks=task_info["max_task"]
            )

            model.eval()
            model.before_task(train_loader, val_loader)    #extend classifier
            #model.before_mtl(ss_train_loader, ss_val_loader)
            model.before_ss(train_loader, val_loader, ss=True, cls_layer=True)
            print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            model.train()
            #model.train_gb(train_loader, val_loader)
            model.train_ss(train_loader, val_loader)
            
            #################### train ssg after classifier #######################################
            # model.before_ss(train_loader, val_loader, ss=True)
            # print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
            # model.train()
            # model.train_gb(train_loader, val_loader)
            # model.train_ss(train_loader, val_loader)
            #######################################################################################

            model.eval()
            model.gb_after_task(inc_dataset)
            

            print("Eval on {}->{}.".format(0, task_info["max_class"]))
            #ypred, ytrue = model.eval_task(test_loader)
            
            # batch size = 1        
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(train_loader, ss=1)            

            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('train classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('train nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("train\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")
            ####################################################################################
            
            ypred, ytrue, ynpred, yntrue, y_top5, yn_top5 = model.eval_task(test_loader, ss=1)          
            
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print(acc_stats)
            results["results"].append(acc_stats)
            memory = model.get_memory()
            ####################################################################################
            #classifier 100 classes
            acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
            print('classifier:     ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_classifier.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            #nearest 100 classes
            acc_stats = utils.compute_accuracy(ynpred, yntrue, task_size=args["increment"])
            print('nearest:        ',acc_stats)

            with open('./results/results_txt/'+args["name"]+'_nearest.txt', "a") as accuracy:
                accuracy.write("test\n")
                for i in acc_stats.values():
                    accuracy.write(str(i) + " ")
                accuracy.write("\n")

            memory = model.get_memory()
            ####################################################################################
    """        
    print(
        "Average Incremental Accuracy: {}.".format(
            results_utils.compute_avg_inc_acc(results["results"])
        )
    )
    """
    #if args["name"]:
    #    results_utils.save_results(results, args["name"])

    del model
    del inc_dataset
    del ss_inc_dataset
    del ltl_inc_dataset
    torch.cuda.empty_cache()


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
