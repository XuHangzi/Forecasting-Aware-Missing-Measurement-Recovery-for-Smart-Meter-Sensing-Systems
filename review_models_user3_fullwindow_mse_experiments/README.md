# Review-model user3 full-window 2013 MSE experiment

This experiment evaluates the trained fine-tuned GAIN models from
`review_training_hyperparam_experiments/models` on 2013 `user_3` full-year
sliding LSTM prediction windows.

It uses the `guided_imputed_processed.csv` files already saved by the review
experiment, so it does not retrain or re-impute.

For the `lambda_3` figure, two additional trained review-style models are used:
`lambda_3=1e-5` and `lambda_3=1e-1`. They can be generated with:

```powershell
conda run -n mytorch python review_models_user3_fullwindow_2013_mse_experiments\train_extra_lambda3_review_models.py
```

## Evaluation windows

- User: `user_3`
- Year: 2013 full year
- Lookback: 720 hours
- Horizon: 168 hours
- Stride: 24 hours
- Windows per hyperparameter setting: 329
- Prediction points per hyperparameter setting: 55272

## Run

From the project root:

```powershell
conda run -n mytorch python review_models_user3_fullwindow_2013_mse_experiments\run_review_models_user3_2013.py
```

## Outputs

- `results/lambda_2_user3_fullwindow_2013.csv`
- `results/lambda_3_user3_fullwindow_2013.csv`
- `figures/lambda_2_user3_fullwindow_2013.jpeg`
- `figures/lambda_3_user3_fullwindow_2013.jpeg`

The two figures are also copied to
`review_training_hyperparam_experiments/figures` without overwriting the
original review figures.
