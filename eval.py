import os
import csv


def check_folder(folder):
    folders1 = os.listdir(folder)
    best_score = 0
    best_para = ""
    for f in folders1:
        file = os.path.join(folder, f, "eval_results.txt")
        if os.path.exists(file):
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("best_eval_accuracy"):
                        parts = line.split(' ')
                        score = float(parts[2])
                        if score > best_score:
                            best_score = score
                            best_para = f
    print(best_score)
    print(best_para)


def get_eval_output(eval_file, pred_label_file, new_file):
    pred_labels = []
    ann_labels = []
    with open(pred_label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("b") and not line.startswith("e") and \
                    not line.startswith("g") and not line.startswith("l"):
                parts = line.split('\t')
                pred_label = parts[2]
                ann_label = parts[1]
                pred_labels.append(pred_label)
                ann_labels.append(ann_label)

    csv_file = open(new_file, 'w')
    field_names = ['id', "context", "question", "answer1", "answer2", "answer3", "answer4", "label", "pred_label"]
    out_anno = csv.DictWriter(csv_file, fieldnames=field_names)
    out_anno.writeheader()

    idx = 0

    with open(eval_file, 'r') as csv_file1:
        reader = csv.DictReader(csv_file1)
        for row in reader:
            ann_label = row["label"]
            assert ann_label == ann_labels[idx]
            row["pred_label"] = pred_labels[idx]
            idx += 1
            out_anno.writerow(row)
    csv_file.close()


if __name__ == "__main__":
    # check_folder("/data/m1/huangl7/CommonsenseQaPlus/baselines/pytorch_pretrained_bert/output/")
    # check_folder("/data/m1/huangl7/CommonsenseQaPlus/baselines/pytorch_pretrained_bert/output1/")
    check_folder("/data/m1/huangl7/CommonsenseQaPlus/baselines/pytorch_pretrained_bert/output2/")
    # get_eval_output("/data/m1/huangl7/CommonsenseQaPlus/baselines/data/test.csv",
    #                 "/data/m1/huangl7/CommonsenseQaPlus/baselines/pytorch_pretrained_bert/output/batch_4_lr_2e-5_epochs5/eval_results.txt",
    #                 "/data/m1/huangl7/CommonsenseQaPlus/baselines/data/test_pred.csv")

