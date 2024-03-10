import re,os
import json 
def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
        text = text.strip()
    return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])

def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i:i + value_len] == value_list:
            found = True
            break
    return found

def trippy_correct(pred, gt):
    # Check Slots
    pred_slot = "-".join(pred.split("-")[:-1])
    gt_slot = "-".join(gt.split("-")[:-1])
    if pred_slot != gt_slot:
        return False
    
    # Check Values
    pred =tokenize(pred.split("-")[-1])
    gt =tokenize(gt.split("-")[-1])
    
    label_maps = json.load(open(os.path.join("data","trippy_utils","label_maps.json"),"r"))

    if pred == gt:
        return True
    elif is_in_list(gt, pred):
        return True
    elif is_in_list(pred, gt):
        return True
    elif gt in label_maps:
        for inform_label_variant in label_maps[gt]:
            if pred == inform_label_variant:
                return True
            elif is_in_list(inform_label_variant, pred):
                return True
            elif is_in_list(pred, inform_label_variant):
                return True
    elif pred in label_maps:
        for value_label_variant in label_maps[pred]:
            if value_label_variant == gt:
                return True
            elif is_in_list(gt, value_label_variant):
                return True
            elif is_in_list(value_label_variant, gt):
                return True
            
    return False

def trippy_correct_old(pred,gt):
    label_maps = json.load(open(os.path.join("data","trippy_utils","label_maps.json"),"r"))
    
    pred ="-".join([ "-".join(pred.split("-")[:-1]),tokenize(pred.split("-")[-1])])
    gt ="-".join([ "-".join(gt.split("-")[:-1]),tokenize(gt.split("-")[-1])])
    if pred == gt:
        return True
    elif gt.split("-")[-1] in label_maps:
        no_match = True
        for variant in label_maps[gt.split("-")[-1]]:
            if "-".join([ "-".join(gt.split("-")[:-1]),variant]) == pred:
                no_match = False
                break
            if no_match:
                return False
            else:
                return True
    else: 
        return False