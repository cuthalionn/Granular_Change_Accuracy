#-----------------------------------
# Run Command
# python compute_accuracy_trade_somdst.py
# Most of the code in this document is based on this repository: https://github.com/SuvodipDey/FGA
#-----------------------------------
from src.gca_trippy import compute_gca_trippy
import os
import json
import pandas as pd
import math
import sys
from src.trippy_utils import trippy_correct


def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data
    
def getBeliefSet(ds):
    bs = set()
    for dom in ds:
        for slot in ds[dom]:
            if ds[dom][slot] != "none":
                t = dom+"-"+slot+"-"+ds[dom][slot]
                bs.add(t)
            else:
                continue
    return bs

def get_unique_slots(pr,gt):
    all_pairs = pr.union(gt)
    slots = ["-".join(pair.split("-")[:-1]) for pair in all_pairs]
    unique_slots = set(slots)
    return len(unique_slots)

# Slot Accuracy
def getSlotAccuracy(gt, pr):
    d1 = trippy_difference(gt,pr)
    d2 = trippy_difference(pr,gt)
    
    s1 = set([d.rsplit("-", 1)[0] for d in d1])
    s2 = set([d.rsplit("-", 1)[0] for d in d2])    
    
    set_i = s1.intersection(s2)
    acc = (30 - len(d1) - len(d2) + len(set_i))/30.0
    return acc

def getRelativeSlotAccuracy(gt,pr):
    # BUG cannot replicate
    d1 = trippy_difference(gt,pr)
    d2 = trippy_difference(pr,gt)
    
    s1 = set([d.rsplit("-", 1)[0] for d in d1])
    s2 = set([d.rsplit("-", 1)[0] for d in d2])    
    
    set_i = s1.intersection(s2)

    T_star = get_unique_slots(pr,gt)

    acc = (T_star - len(d1) - len(d2) + len(set_i))/T_star if T_star!= 0 else 0 
    
    # acc = (T_star - M - W) / T_star if T_star!= 0 else 0
    return acc

# Slot Accuracy Computation taken from TRADE model
def compute_acc(gold, pred):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = 30
    ACC = 30 - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

# Average Goal Accuracy
def getAvgGoalAccuracy(gt, pr):
    set_i = trippy_intersection(gt,pr)
    acc = -1
    if(len(gt)>0):
        acc = len(set_i)/float(len(gt))
    return acc

def getImprovedAvgGoalAccuracy(gt, pr):    
    set_i = trippy_intersection(gt,pr)
    acc = -1
    if(len(gt.union(pr))>0):
        acc = len(set_i)/float(len(gt.union(pr)))
    return acc

# Flexible Goal Accuracy
def getFGA(gt_list, pr_list, turn_diff, L):
    gt = gt_list[-1]
    pr = pr_list[-1]
    diff1 = trippy_symmetric_difference(gt,pr)
    # diff1 = gt.symmetric_difference(pr)
    if len(diff1)==0: #Exact match
        return 1
    else:
        if len(gt_list)==1: 
            #Type 1 error
            #First turn is wrong
            return 0
        else:
            diff2 = trippy_symmetric_difference(gt_list[-2],pr_list[-2])
            # diff2 = gt_list[-2].symmetric_difference(pr_list[-2])
            if len(diff2)==0: 
                #Type 1 error
                #Last turn was correct i.e the error in current turn
                return 0
            else:
                tgt = trippy_difference(gt,gt_list[-2])
                # tgt = gt.difference(gt_list[-2])
                tpr = trippy_difference(pr,pr_list[-2])
                # tpr = pr.difference(pr_list[-2])
                if(not trippy_issubset(tgt,pr) or not  trippy_issubset(tpr,gt)): 
                    #Type 1 error
                    #There exists some undetected/false positive intent in the current prediction
                    return 0
                else:
                    #Type 2 error
                    #Current turn is correct but source of the error is some previous turn
                    return (1-math.exp(-L*turn_diff))
    
def getModifiedBS(bs):
    bs_new = {}
    for k in bs:
        bs_new[k] = {}
        for slot in bs[k]:
            sl = slot
            v = bs[k][slot]
            if "book" in slot:
                sl = slot.split(' ')[1]
            bs_new[k][sl] = v
    return bs_new

def trippy_issubset(set1,set2):
    is_subset = True
    for item in set1:
        is_in = False
        for item2 in set2:
            if trippy_correct(item,item2):
                is_in = True
                break
        if not is_in:
            is_subset = False
            break
    return is_subset

def trippy_intersection(set1,set2):
    intersect = set()
    for item in set1:
        is_in = False
        
        for item2 in set2:
            if trippy_correct(item,item2):
                is_in = True
                break
        if is_in:
            intersect.add(item)
                
    return intersect

def trippy_difference(set1,set2):
    diff = set()
    
    for item in set1: 
        is_in = False
        for item2 in set2:
             if trippy_correct(item,item2):
                 is_in = True
                 break
        if not is_in:
            diff.add(item)
    
    return diff

def trippy_symmetric_difference(gts,prs):
    diff = set()
    
    for gt in gts: 
        is_in = False
        for pr in prs:
             if trippy_correct(pr,gt):
                 is_in = True
                 break
        if not is_in:
            diff.add(gt)
            
    for pr in prs: 
        is_in = False
        for gt in gts:
             if trippy_correct(pr,gt):
                 is_in = True
                 break
        if not is_in:
            diff.add(pr)
            
    return diff
    

def getModelAccuracy(dst_res_path, dialog_data):
    # dst_res_path = os.path.join(model_name, model_name+"_result.json")
    dst_res = loadJson(dst_res_path)
    
    joint_acc = 0
    slot_acc = 0
    relative_slot_acc = 0
    avgGoalAcc = []
    improvedAvdGoalAcc = []
    fga = [0, 0, 0, 0]
    turn_acc = 0
    total = 0
    lst_lambda = [0.25, 0.5, 0.75, 1.0]
    dial_metrics = {}
    for idx in dst_res:
        res = dst_res[idx]
        log = dialog_data[idx]['log']
        sys = " "
        
        gt_list = []
        pr_list = []
        error_turn = -1
        fga_dial = [0, 0, 0, 0]
        total_dial = 0
        for turn in res:
            total_dial += 1
            total+=1
            i = 2*int(turn)
            usr = log[i]['text'].strip()
            if(i>0):
                sys = log[i-1]['text'].strip()

            gt = getBeliefSet(res[turn]['gt'])
            pr = getBeliefSet(res[turn]['pr'])

            gt_list.append(gt)
            pr_list.append(pr)

            #print(f"Sys_{turn}: {sys}")
            #print(f"Usr_{turn}: {usr}")
            #print(f"GT_{turn}: {getModifiedBS(res[turn]['gt'])}")
            #print(f"PR_{turn}: {getModifiedBS(res[turn]['pr'])}")
            #print("-"*40)

            # diff1 = gt.symmetric_difference(pr)
            diff = trippy_symmetric_difference(gt,pr)
            m = 1 if len(diff)==0 else 0
            joint_acc+=m

            sa = getSlotAccuracy(gt, pr)
            slot_acc+=sa

            relative_sa = getRelativeSlotAccuracy(gt, pr)
            relative_slot_acc += relative_sa

            aga = getAvgGoalAccuracy(gt, pr)
            if(aga>=0):
                avgGoalAcc.append(aga)

            improved_aga = getImprovedAvgGoalAccuracy(gt, pr)
            if(improved_aga>=0):
                improvedAvdGoalAcc.append(improved_aga)
            
            m = 0
            for l in range(len(lst_lambda)):
                m = getFGA(gt_list, pr_list, int(turn)-error_turn, lst_lambda[l])
                fga[l]+=m
                fga_dial[l] += m
            if(m==0):
                error_turn = int(turn)
            else:
                turn_acc+=1

        for l in range(len(lst_lambda)):
            fga_dial[l] = round(fga_dial[l]*100.0/total_dial,2)
        dial_metrics[idx] = fga_dial

    # print(f"Total: {total}, Exact Match: {joint_acc}, Turn Match: {turn_acc}")
    joint_acc = round(joint_acc*100.0/total,2)
    slot_acc = round(slot_acc*100.0/total,2)
    relative_slot_acc = round(relative_slot_acc*100.0/total,2)
    avg_goal_acc = round(sum(avgGoalAcc)*100.0/len(avgGoalAcc),2)
    improved_avg_goal_acc = round(sum(improvedAvdGoalAcc)*100.0/len(avgGoalAcc),2)
    print(f"Joint Acc = {joint_acc},\
            Slot Acc = {slot_acc},\
            Avg. Goal Acc = {avg_goal_acc},\
            Improved Avg. Goal Acc = {improved_avg_goal_acc},\
            Relative Slot Acc = {relative_slot_acc}")
    for l in range(len(lst_lambda)):
        fga_acc = round(fga[l]*100.0/total,2)
        print(f"FGA with L={lst_lambda[l]} : {fga_acc}")
    return dial_metrics
#-----------------------------------
def main():
    #Load raw data

    dialog_data_file = os.path.join("data",'data.json')
    dialog_data = loadJson(dialog_data_file)
    file_name = "trippy"
    folder_name = os.path.join("data", file_name)
    eval_file_dir = os.listdir(folder_name)
    for i,file in enumerate(eval_file_dir):
        print("-"*40)
        print("Results for {} :-".format(file))
        getModelAccuracy(os.path.join(folder_name,file),  dialog_data)

        data = json.load(open(os.path.join(folder_name,file),'r'))
        gca_metric,_,_, _ = compute_gca_trippy(data)
        print("GCA: {:.2f}".format(gca_metric * 100))
        # print(mr)
if __name__ == "__main__":
    main()
# -----------------------------------