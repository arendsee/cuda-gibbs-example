# Bladder Cancer sample dataset

This sample dataset can be found [here](https://ratecalc.cancer.gov/).

`full-dataset.csv` contains the entire dataset, including columns the Gibbs
sampler does not use. The `prepared-data.csv` is appropriate for usage in the
Gibbs sampler. It was produced from `full-dataset.csv` with the following command:

```bash
awk 'BEGIN{FS=","} NR > 2 {print $6/100000, $7}' full-dataset.csv > prepared-data.csv
```

Column 1 contains person years (in units of 100,000), column 2 contains
mortality rate.

Here is a description of the model, borrowed directly from Jarad Niemi's
STAT544 course at Iowa State University.

Let
```
y_i     be the number of observed deaths in county i
n_i     be the number of person-years in county i divided by 100,000
theta_i is the expected deaths per 100,000 person-years
```

Assume

```
y_i ∼ Pois(n_i, theta_i)
theta_i ∼ Ga(a, b)
alpha ∼ Unif (0, alpha_0 )
beta ∼ Unif (0, beta_0 )
```

with `alpha` and `beta` independent and for some chosen values of `a_0` and `b_0`.
