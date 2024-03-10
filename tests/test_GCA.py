import pytest
import json
from src.gca import compute_gca
from src.compute_accuracies import getRelativeSlotAccuracy, getBeliefSet

@pytest.fixture
def sample_data():
    return [json.load(open("data/poc/poc_result1.json", "r")),
            json.load(open("data/poc/poc_result2.json", "r")),
            json.load(open("data/poc/poc_result3.json", "r")),
            json.load(open("data/poc/poc_result4.json", "r")),
            json.load(open("data/poc/poc_result5.json", "r"))
            ]

def test_gca(sample_data):
    gca_metric1,_,_,_ = compute_gca(sample_data[0])
    gca_metric2,_,_,_ = compute_gca(sample_data[1])
    gca_metric3,_,_,_ = compute_gca(sample_data[2])
    gca_metric4,_,_,_ = compute_gca(sample_data[3])
    gca_metric5,_,_,_ = compute_gca(sample_data[4])
    
    assert (gca_metric1 == pytest.approx(0.83,0.05))
    assert (gca_metric2 == pytest.approx(0.1,0.05))
    assert (gca_metric3 == pytest.approx(0.34,0.05))
    assert (gca_metric4 == pytest.approx(0.76,0.05))
    assert (gca_metric5 == pytest.approx(0.47,0.05))

def test_rsa(sample_data):
    accuracies = []
    for dial in sample_data:
        dial = dial["MUL0003.json"]
        rsa = 0
        total = 0
        for turn_id in dial:
            total+=1
            gt = getBeliefSet(dial[turn_id]["gt"])
            pr = getBeliefSet(dial[turn_id]["pr"])
            rsa += getRelativeSlotAccuracy(gt,pr)
        rsa = round(rsa*100.0/total,2)
        accuracies.append(rsa)
    
    assert (accuracies[0] == pytest.approx(27.27,0.05))
    assert (accuracies[1] == pytest.approx(53.03,0.05))
    assert (accuracies[2] == pytest.approx(16.66,0.05))
    assert (accuracies[3] == pytest.approx(62.59,0.05))
    assert (accuracies[4] == pytest.approx(25.19,0.05))