

def evaluate(type, golds, preds, supertag_idx):
    assert len(golds) == len(preds)
    if type == "acc":
        correct = 0.0
        for _ in range(len(golds)):
            gold = golds[_]
            gold_str = " ".join(gold).strip()

            pred = preds[_]
            pred_str = " ".join(pred).strip()

            if gold_str == pred_str:
                correct += 1.0
        return correct/len(preds)

    elif type == "word_acc":
        correct = 0.0
        sum = 0.0
        for _ in range(len(golds)):
            gold = golds[_].split(" ")
            pred = preds[_].split(" ")

            for idx in range(len(gold)):
                if gold[idx] not in supertag_idx:
                    continue
                sum += 1.0
                if idx < len(pred) and gold[idx] == pred[idx]:
                    correct += 1.0
        return correct / sum