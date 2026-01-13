data {
    int<lower=1> J;
    vector[J] y1_bar;
    vector[J] y0_bar;
    vector<lower=0>[J] sigma_y1_bar;
    vector<lower=0>[J] sigma_y0_bar;
    array[J] int<lower=1, upper=J> j;
}
parameters {
    real b;
    real mu_theta;
    real<lower=0> sigma_theta;
    vector[J] eta_theta;
}
transformed parameters {
    vector[J] theta;

    theta = mu_theta + sigma_theta * eta_theta;
}
model {
    eta_theta ~ normal(0, 1);

    y1_bar ~ normal(b + theta, sigma_y1_bar);
    y0_bar ~ normal(b, sigma_y0_bar);
}