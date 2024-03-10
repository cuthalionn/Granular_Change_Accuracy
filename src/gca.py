from pickle import NONE
import sys
import random
import json
import os 
from collections import defaultdict
random.seed(10)
LINE_SEP = "-------------------------------------------\n"
SECTION_SEP = "===================================\n\
===================================\n"

def weighted_harmonic_mean(weights, items):
    """ Calculate a weighted harmonic over a list of items.

    Args:
        weights (double): weights list
        items (double): item list

    Returns:
        double: harmonic mean
    """

    return sum(weights) / (sum([weights[i]/items[i] for i in range(len(items))]))

def gca(weight_multipliers,weights, values):
    """ Calculate GCA.

    Args:
        weight_multipliers (double): multiplier list
        weights (double): weights list
        values (double): value list

    Returns:
        double: GCA value
    """

    multiplied_weights = [weight_multipliers[i]*weights[i] for i in range(len(weights))]
    return weighted_harmonic_mean(multiplied_weights, values)

class ModelResult():
    """ Class representing a model evaluation result.

        This is used as a container class to store all values related to evaluation 
        in an organized way.
    """

    def __init__(self,
                all_turn_beliefs,
                all_preds_beliefs,
                wrong_preds,
                total_turn,
                total_pred,
                total_ground_truth,
                wrong_pred,
                correct_pred,
                precision,
                recall,
                F1,
                missed_label,
                missed_label_ratio,
                over_pred_label,
                over_pred_label_ratio,
                label_hit_ratio,
                gca_metric,
                turn_mistakes,
                model_name="unnamed") -> None:

        self.all_turn_beliefs = all_turn_beliefs
        self.all_preds_beliefs = all_preds_beliefs
        self.wrong_preds = wrong_preds
        self.total_turn = total_turn
        self.name = model_name
        self.total_pred = total_pred
        self.total_ground_truth = total_ground_truth
        self.wrong_pred = wrong_pred
        self.correct_pred = correct_pred
        self.precision = precision
        self.recall = recall
        self.F1 = F1
        self.missed_label = missed_label
        self.missed_label_ratio = missed_label_ratio
        self.over_pred_label = over_pred_label
        self.over_pred_label_ratio=over_pred_label_ratio
        self.gca_metric = gca_metric
        self.label_hit_ratio= label_hit_ratio
        self.turn_mistakes = turn_mistakes

    def __str__(self):

        result = ["Statistics for model @ {}:",
                  "{}"
                 "Total number of predictions: {}",
                 "Total number of ground truth values: {}",
                 "Total number of wrong predictions: {} (P: {:.2f}, R: {:.2f}, F1: {:.2f})",
                 "Total number of correct predictions: {} ",
                 "Total number of missed labels: {} ({:.2f}%)",
                 "Total number of over-predictions: {} ({:.2f}%)",
                 "Label hit ratio: {:.2f} ",
                 "GCA: {:.2f}",
                 "-------------------------------------------"
                 ]

        result = "\n".join(result).format(self.name,
                                        LINE_SEP,
                                        self.total_pred,
                                        self.total_ground_truth,
                                        self.wrong_pred,
                                        self.precision,
                                        self.recall,
                                        self.F1,
                                        self.correct_pred,
                                        self.missed_label,
                                        self.missed_label_ratio,
                                        self.over_pred_label,
                                        self.over_pred_label_ratio,
                                        self.label_hit_ratio,
                                        self.gca_metric)

        return result

def get_belief_dict(belief):
    """ Convert dialogue state from list to dictionary format.

    Args:
        belief (list): dialogue state as a list of predictions

    Returns:
        dict: dialogue state as a dictionary with mapping -> slot labels: slot values
    """

    belief_dict = {}

    for domain, sub_dict in belief.items():
        for k,v in sub_dict.items():
            domain_key = f"{domain}_{k}"
            belief_dict[domain_key] = v
    return belief_dict

def calculate_gca(wrong,correct,overpred,missed):
    """ Calculate GCA given four fundamental evaluation metrics

    Args: 
        wrong (int): Number of wrong predictions
        correct (int): Number of correct predictions
        overpred (int): Number of over-predictions
        missed (int): Number of missed-predictions

    Returns:
        double: GCA score
    """

    prec_denom = (wrong + correct + overpred)
    recall_denom = (wrong + correct + missed)
    label_prec = (correct + wrong) / prec_denom if prec_denom > 0 else 0
    label_recall = (correct + wrong) / recall_denom if recall_denom > 0 else 0
    lhr_denom = wrong + correct + overpred+missed
    prec = correct / prec_denom if prec_denom > 0 else 0
    recall = correct / recall_denom if recall_denom > 0 else 0
    labelhit_ratio = (correct + wrong) / lhr_denom if lhr_denom > 0 else 0
    missed_label_ratio = (missed / recall_denom) * 100 if recall_denom > 0 else 0
    over_pred_label_ratio = (overpred / prec_denom) * 100 if prec_denom > 0 else 0
    F1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

    if prec>0 and recall> 0 and label_prec>0 and label_recall >0:
        #TODO: Make this alpha into an argument or a global variable
        # Then reflect this change to other documents.
        alpha = 1 - 1/11
        gca_metric = gca([alpha,alpha,1-alpha,1-alpha],
                        [prec_denom,recall_denom,prec_denom,recall_denom],
                        [prec,recall,label_prec,label_recall])
    else:
        gca_metric = 0

    return gca_metric,prec,recall,labelhit_ratio, missed_label_ratio, over_pred_label_ratio,F1

def compute_gca(data):
    """Computes GCA Scores given the DST results data

    Args:
        data (dict): DST result data in dictionary format

    Returns:
    gca_metric,all_dial_metrics,all_dial_mrs, mr 
        double: GCA metric
        list: List of GCA metrics (for each dialogue)
        list: list of Model Results (for each dialogue)
        Model Result: Model Result (pertaining to overall model)
    """


    total_pred, total_ground_truth, wrong_pred, correct_pred = 0, 0, 0 ,0
    missed_label, over_pred_label, total_turn = 0 , 0 , 0
    wrong_preds = defaultdict(lambda:[])
    all_preds_beliefs = defaultdict(lambda:{})
    all_turn_beliefs = defaultdict(lambda:{})
    all_dial_metrics = {}
    all_dial_mrs = {}
    dial_turn_mistakes = {}

    for dial_key in data.keys():
        total_turn += len(list(data[dial_key].keys()))
        past_turn_belief = {}
        past_pred_belief = {}

        active_gt_slot_labels = set()
        active_pred_slot_labels = set()

        dial_correct, dial_wrong, dial_missed, dial_overpred = 0, 0, 0, 0
        turn_mistakes = []

        for turn_id in data[dial_key].keys():
            turn_mistake = 0
            correct_labels = set()
            wrong_labels = set()
            gt = data[dial_key][turn_id]["gt"]
            pr = data[dial_key][turn_id]["pr"]

            turn_belief = get_belief_dict(gt)
            pred_belief = get_belief_dict(pr)

            for k in turn_belief:
                active_gt_slot_labels.add(k)

            for k in pred_belief:
                active_pred_slot_labels.add(k)

            dropped_gt_labels = (active_gt_slot_labels - set(turn_belief.keys()))
            for label in dropped_gt_labels:
                turn_belief[label]="none"

            dropped_pred_labels = (active_pred_slot_labels - set(pred_belief.keys()))
            for label in dropped_pred_labels:
                pred_belief[label]="none"

            turn_dif = dict(set(turn_belief.items()) - set(past_turn_belief.items()))
            pred_dif = dict(set(pred_belief.items()) - set(past_pred_belief.items()))

            all_turn_beliefs[dial_key][turn_id]=turn_belief
            all_preds_beliefs[dial_key][turn_id]=pred_belief

            if turn_belief != pred_belief:
                wrong_preds[dial_key].append(turn_id)
            
            total_ground_truth += len(turn_dif.keys())
            total_pred += len(pred_dif.keys())  

            for label in turn_dif:
                if label not in pred_belief and turn_dif[label] != "none":
                    dial_missed += 1
                    turn_mistake += 1
                elif label not in pred_belief and turn_dif[label] == "none":
                    dial_correct += 1
                    correct_labels.add(label)
                elif pred_belief[label] == turn_dif[label]:
                    dial_correct += 1
                    correct_labels.add(label)
                elif turn_dif[label]=="none":
                    dial_overpred += 1
                    turn_mistake += 1 
                elif pred_belief[label] != turn_dif[label]:
                    dial_wrong += 1
                    turn_mistake += 1 
                    wrong_labels.add(label)
                else:
                    continue

            for label in pred_dif:
                if label not in turn_belief and pred_dif[label] != "none":
                    dial_overpred += 1 
                    turn_mistake += 1 
                elif label not in turn_belief and pred_dif[label] == "none":
                    dial_correct += 1 
                elif pred_dif[label] == turn_belief[label] and label not in correct_labels:
                    dial_correct += 1
                elif pred_dif[label] == "none":
                    dial_missed += 1
                    turn_mistake += 1 
                elif pred_dif[label] != turn_belief[label] and label not in wrong_labels:
                    dial_wrong += 1
                    turn_mistake += 1 
                else:
                    continue
            turn_mistakes.append(turn_mistake)
            past_turn_belief = turn_belief
            past_pred_belief = pred_belief

        # Dial metric
        dial_gca,dial_prec,dial_recall,dial_labelhit_ratio,_,_,_ = calculate_gca(dial_wrong,
                                                                                dial_correct,
                                                                                dial_overpred,
                                                                                dial_missed)
        all_dial_metrics[dial_key] = dial_gca
        mr_dial = ModelResult(-1,
                            -1,
                            dial_wrong,
                            -1,
                            -1,
                            -1,
                            dial_wrong,
                            dial_correct,
                            dial_prec,
                            dial_recall,
                            -1,
                            dial_missed,
                            -1,
                            dial_overpred,
                            -1,
                            dial_labelhit_ratio,
                            all_dial_metrics[dial_key],
                            turn_mistakes,
                            model_name="none")
        all_dial_mrs[dial_key] = mr_dial
        dial_turn_mistakes[dial_key] = turn_mistakes

        #Update Data level metrics
        missed_label += dial_missed
        correct_pred += dial_correct
        over_pred_label += dial_overpred
        wrong_pred += dial_wrong

    ####
    gca_metric,prec,recall,label_hit_ratio,missed_label_ratio,over_pred_label_ratio,F1 = calculate_gca(wrong_pred,
                                                                                                      correct_pred,
                                                                                                      over_pred_label,
                                                                                                      missed_label)

    model_result = ModelResult(all_turn_beliefs,
                     all_preds_beliefs,
                     wrong_preds,
                     total_turn,
                     total_pred,
                     total_ground_truth,
                     wrong_pred,
                     correct_pred,
                     prec,
                     recall,
                     F1,
                     missed_label,
                     missed_label_ratio,
                     over_pred_label,
                     over_pred_label_ratio,
                     label_hit_ratio,
                     gca_metric,
                     dial_turn_mistakes)

    return gca_metric,all_dial_metrics,all_dial_mrs, model_result

def main():
    """ Main function.
    """

    file_name = sys.argv[1]
    folder_name = os.path.join("data",file_name)
    eval_file_dir = os.listdir(folder_name)
    for data_path in eval_file_dir:

        print("-"*40)
        print(f"Proof of concept for data {data_path} :-")
        data = json.load(open(os.path.join(folder_name,data_path), encoding="utf-8"))
        _,_,_, model_result = compute_gca(data)
        print(model_result)

if __name__ == "__main__":
    main()
