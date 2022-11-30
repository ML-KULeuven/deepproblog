# Hand-written formula experiment
This experiment uses the hand-written formula dataset, introduced in *Li et al., "Closed Loop Neural-Symbolic Learning via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning"*.
To download the data, run the `download_hwf.sh` script in the `data/` directory. This requires the gdown command, which can be installed using `pip install gdown`.
Alternatively, download the data manually, and make sure it has the following structure:
```
HWF/
    data/
        expr_train.json
        expr_test.json
        Handwritten_Math_Symbols/
            +/
            -/
            div/
            times/
            0/
            ...
            9/
```

