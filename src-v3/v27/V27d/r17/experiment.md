[E]完成V27d-r17冻结TR-only迁移与退化审计实验
为了求证冻结的V27d-r15 L1 treatment与V27c-r23 L2 robust_control能否在OriData Excel的TR-only数据上运行，使用冻结端点进行审计并尝试单seed TR-only退化L2训练；证据包括config、manifest、contract smoke和evaluation JSON。
实验结论：迁移链路完成，但RD不可用、test未创建、L3 unavailable；冻结L2验证Accuracy 56.47%、Macro-F1 36.61%，退化模型验证Accuracy 64.71%、Macro-F1 58.95%，不作泛化或部署结论。
