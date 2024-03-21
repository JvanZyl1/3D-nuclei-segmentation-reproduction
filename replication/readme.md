#### Todos 
> Add the things you already did here too so we can keep track of them. 

So far done:

- Dataset class in Pytorch: some preprocessing steps and normalization but we need interpolation and padding
- NSN network: correct architecture and training loop but we need to train it and save the weights
- NDN network: correct architecture and training loop but it shrinks the depth too much at deeper layers (will be fixed with padding)
- Metrics: implemented but not tested

To do:

- [ ] Preprocessing steps to the image (interpolation and mirror padding)
- [x] (Aral; done but not tested) All the metrics described in metrics.py 
- [ ] Water-based marker shed for instance segmentation
- [x] (Dani) Visualizing the training and test error per epoch
- [ ] Visualizing inference on cells
- [ ] Quantitative observations about the inferred cells
- [ ] Training early stopping criterion: "The trained model with the highest IoU in cross-validation was used to analyse the test dataset" (p. 2). 

#### TA Meeting
Deliverables 
- Blog post indicating approach, difficulties in reproducing, comparison with paper results; and explain differences. 
- Poster presentation

- Contact supervisor about data format; need a few labelled ground truths as well probably. 

#### Planning
Four Criteria:

- **Replicate**: A full implementation from scratch without using any pre-existing code? (does looking at an existing implementation count?)
- **New Code Variant**: Rewrote or ported existing code to be more efficient/readable. (does reimplementing in `pytorch` count?)
- **New Data**: Evaluate different datasets to obtain similar results. (part of the assignment: replicate for new cell polarity data)

###### Pick one
- **Reproduce?**: Existing code was evaluated. (is it fine to just check the code quality, or do we have to find the original model weights and run the entire thing?)
- **Hyperparams Check**: Evaluating sensitivity to hyperparameters. 

###### Probably Not
- **New algorithm variant**: Evaluating a slightly different variant (only Dani seems to know alternatives)
- **Ablation study**: Additional ablation studies. (both NDN and NSN are dependent on each other, so not sure how to proceed.)
