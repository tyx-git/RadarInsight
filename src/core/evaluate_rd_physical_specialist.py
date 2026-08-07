from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.rd_physical_specialist_dataset import RD_FIVE_LABELS, PhysicalRDSpecialistWindowDataset, collate_physical_rd_specialist_windows, specialist_label_names
from .rd_physical_specialist_artifacts import load_physical_rd_specialist_checkpoint
from .rd_physical_specialist_trainer import _route_recall_by_original_class
from .rd_physical_parent_aggregation import mean_parent_probabilities


def _metrics(logits: torch.Tensor, labels: torch.Tensor, names: tuple[str, ...]) -> dict[str, object]:
    matrix = [[0 for _ in names] for _ in names]
    for truth, pred in zip(labels.tolist(), logits.argmax(-1).tolist()): matrix[int(truth)][int(pred)] += 1
    ps=[]; rs=[]; fs=[]; per={}
    for i, name in enumerate(names):
        tp=matrix[i][i]; fp=sum(matrix[r][i] for r in range(len(names)) if r != i); fn=sum(matrix[i][c] for c in range(len(names)) if c != i); p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); f=2*p*r/max(1e-12,p+r); support=sum(matrix[i]); per[name]={"precision":p,"recall":r,"f1":f,"support":support,"support_stability":"low" if 0 < support < 30 else ("no_support" if not support else "adequate")}; ps.append(p); rs.append(r); fs.append(f)
    directional = {f"{names[truth]}_to_{names[prediction]}": matrix[truth][prediction] for truth in range(len(names)) for prediction in range(len(names)) if truth != prediction}
    return {"accuracy":float((logits.argmax(-1)==labels).float().mean()) if labels.numel() else 0.0,"macro_precision":sum(ps)/len(ps) if ps else 0.0,"macro_recall":sum(rs)/len(rs) if rs else 0.0,"macro_f1":sum(fs)/len(fs) if fs else 0.0,"balanced_accuracy":sum(rs)/len(rs) if rs else 0.0,"sample_count":int(labels.numel()),"per_class":per,"confusion_matrix":{"labels":list(names),"matrix":matrix},"directional_errors":directional}


def _parent_metrics(logits: torch.Tensor, labels: torch.Tensor, parents: list[str], names: tuple[str, ...]) -> dict[str, object]:
    parent_probabilities, parent_labels, parent_count = mean_parent_probabilities(logits, labels, parents)
    return {"parent_count": parent_count, "aggregation": "arithmetic mean of per-window probabilities, then argmax", "metrics": _metrics(parent_probabilities, parent_labels, names)}


def evaluate_rd_physical_specialist(checkpoint: Path, output: Path, *, split: str = "val", device_name: str = "cuda", max_batches: int = 0) -> dict[str, object]:
    if split != "val": raise ValueError("RD specialist evaluator permits val only; test is disabled")
    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"): raise RuntimeError("RD specialist evaluator requires CUDA")
    device=torch.device(device_name); model, config, artifacts, checkpoint_payload=load_physical_rd_specialist_checkpoint(checkpoint,Path.cwd(),device); names=specialist_label_names(config.kind)
    dataset=PhysicalRDSpecialistWindowDataset(Path.cwd()/"data",kind=config.kind,index_name=config.index_name,split_name="val",normalization_stats_name=config.normalization_stats_name); loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=False,num_workers=0,collate_fn=collate_physical_rd_specialist_windows)
    logits_rows=[]; labels_rows=[]; parents=[]; predictions=[]
    with torch.no_grad():
        for batch_index,batch in enumerate(loader,1):
            if max_batches and batch_index>max_batches: break
            out=model(batch.features.to(device),batch.time_seconds.to(device),batch.padding_mask.to(device),batch.quality_features.to(device),batch.range_offset_m.to(device),batch.velocity_mps.to(device),batch.raw_amplitude_db.to(device)); logits=out.logits.detach().cpu(); labels=batch.labels.detach().cpu(); logits_rows.append(logits); labels_rows.append(labels); parents.extend(batch.parent_sample_ids)
            for i,sample in enumerate(batch.sample_ids): predictions.append({"sample_id":str(sample),"parent_sample_id":str(batch.parent_sample_ids[i]),"label_name_true":str(batch.label_names[i]),"label_name_pred":names[int(logits[i].argmax())],"probabilities":[float(x) for x in torch.softmax(logits[i],-1)]})
    logits=torch.cat(logits_rows) if logits_rows else torch.empty((0,len(names))); labels=torch.cat(labels_rows) if labels_rows else torch.empty((0,),dtype=torch.long); full=max_batches==0 and len(predictions)==len(dataset)
    payload={"scope":f"physical RD-only {config.kind} specialist validation; no TR input, fusion, reject tuning, or test evaluation" if full else "restricted RD specialist smoke validation; incomplete split, no performance claim","kind":config.kind,"labels":list(names),"checkpoint":checkpoint.as_posix(),"checkpoint_epoch":int(checkpoint_payload["epoch"]),"metrics":_metrics(logits,labels,names),"route_recall_by_original_class":_route_recall_by_original_class(logits,[str(row["label_name_true"]) for row in predictions],kind=config.kind),"parent_metrics":_parent_metrics(logits,labels,parents,names),"evaluation_completion":{"full_split_evaluated":full,"dataset_window_count":len(dataset),"evaluated_window_count":len(predictions),"excluded_by_batch_limit_count":len(dataset)-len(predictions),"max_batches":max_batches},"source":{"available":True,"value":"cq08_track"},"session":{"available":False,"reason":"physical index has no session field"},"online_stable_decision":{"available":False,"reason":"no production stable-decision state machine connected"},"rejection_capability":{"supports_rejection":False,"unknown_target_is_class":config.kind=="cu","abstain_output":False,"threshold":None},"test_evaluation_executed":False,"artifact_manifest":artifacts["manifest"]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); pd.DataFrame(predictions).to_csv(output.with_suffix(".predictions.csv"),index=False); return payload


def main() -> None:
    parser=argparse.ArgumentParser(description="Evaluate an RD specialist checkpoint on validation only."); parser.add_argument("--checkpoint",required=True); parser.add_argument("--output",required=True); parser.add_argument("--split",default="val"); parser.add_argument("--device",default="cuda"); parser.add_argument("--max-batches",type=int,default=0); args=parser.parse_args(); result=evaluate_rd_physical_specialist(Path(args.checkpoint),Path(args.output),split=args.split,device_name=args.device,max_batches=args.max_batches); print(json.dumps({k:v for k,v in result.items() if k != "predictions"},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
