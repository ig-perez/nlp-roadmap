# Results and TODOs

- [ ] WRITE ABOUT WORKING WITH TMUX!

## Part C results
1. train the vanilla model w/out pretraning through the finetuning parameter

    command: python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv
    results:
        - epochs: 75
        - training loss: 0.21965
        - lr 5.100024e-04

2. evaluate model on the dev set

    command: python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions
    results:
        - data has 418352 characters, 256 unique.
        - number of parameters: 3323392
        - 500it [01:02,  8.20it/s]
        - Correct: 7.0 out of 500.0: 1.4000000000000001%

3. evaluate model on the test set. No score since the test set don't have true labels. The predictions are saved in --outputs_path

   command: python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions

4. London baseline (the accuracy if the model would predict London for each line of the dev set): Correct: 25.0 out of 500.0: 5.0%