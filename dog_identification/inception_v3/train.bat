python G:/workdoc/TS/dog_identification/inception_v3/retrain.py ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 1000 ^
--model_dir G:/workdoc/TS/dog_identification/inception_v3/inception_model/ ^
--output_graph output/out_graph.pb ^
--output_labels output/output_labels.csv ^
--image_dir G:/workdoc/TS/dog_identification/data/train_classified/
pause