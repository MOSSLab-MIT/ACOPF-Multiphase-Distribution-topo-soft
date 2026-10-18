# Penalty FS vs Ipopt error summary

Seed `1`, feasible test set. Errors are elementwise over all samples × DER phases / non-slack nodes.

| Grid | Stage | Quantity | MAE | RMSE | Max abs |
|---|---|---|---:|---:|---:|
| 13-bus | before SLFS | Active DER $p$ | 1.6134e-03 | 2.3106e-03 | 1.5572e-02 |
| 13-bus | before SLFS | Reactive DER $q$ | 2.3962e-03 | 4.1386e-03 | 4.4656e-02 |
| 13-bus | before SLFS | Voltage magnitude $|V|$ | 1.8355e-03 | 3.2511e-03 | 3.6220e-02 |
| 13-bus | Penalty+SLFS | Active DER $p$ | 1.6306e-03 | 2.3576e-03 | 2.3173e-02 |
| 13-bus | Penalty+SLFS | Reactive DER $q$ | 2.4065e-03 | 4.1502e-03 | 4.4656e-02 |
| 13-bus | Penalty+SLFS | Voltage magnitude $|V|$ | 1.8112e-03 | 3.2440e-03 | 3.6220e-02 |
| 123-bus | before SLFS | Active DER $p$ | 7.1767e-04 | 1.3006e-03 | 9.9933e-03 |
| 123-bus | before SLFS | Reactive DER $q$ | 1.4290e-03 | 2.7817e-03 | 2.5046e-02 |
| 123-bus | before SLFS | Voltage magnitude $|V|$ | 1.4206e-03 | 2.1592e-03 | 2.1529e-02 |
| 123-bus | Penalty+SLFS | Active DER $p$ | 7.2272e-04 | 1.3143e-03 | 1.3333e-02 |
| 123-bus | Penalty+SLFS | Reactive DER $q$ | 1.4372e-03 | 2.7861e-03 | 2.5046e-02 |
| 123-bus | Penalty+SLFS | Voltage magnitude $|V|$ | 1.4183e-03 | 2.1806e-03 | 2.1529e-02 |
| 240-bus | before SLFS | Active DER $p$ | 6.0162e-05 | 1.2623e-04 | 2.7436e-03 |
| 240-bus | before SLFS | Reactive DER $q$ | 2.9542e-04 | 5.3537e-04 | 3.7012e-03 |
| 240-bus | before SLFS | Voltage magnitude $|V|$ | 3.0820e-03 | 4.0708e-03 | 2.6616e-02 |
| 240-bus | Penalty+SLFS | Active DER $p$ | 5.7495e-05 | 1.2136e-04 | 2.3000e-03 |
| 240-bus | Penalty+SLFS | Reactive DER $q$ | 2.9539e-04 | 5.3479e-04 | 3.7012e-03 |
| 240-bus | Penalty+SLFS | Voltage magnitude $|V|$ | 3.0675e-03 | 4.0630e-03 | 2.6616e-02 |
| 906-bus | before SLFS | Active DER $p$ | 8.8993e-06 | 1.3288e-04 | 4.9627e-03 |
| 906-bus | before SLFS | Reactive DER $q$ | 6.7025e-04 | 1.5660e-03 | 9.9988e-03 |
| 906-bus | before SLFS | Voltage magnitude $|V|$ | 7.1488e-04 | 1.1350e-03 | 2.1251e-02 |
| 906-bus | Penalty+SLFS | Active DER $p$ | 1.0356e-05 | 9.7712e-05 | 3.4314e-03 |
| 906-bus | Penalty+SLFS | Reactive DER $q$ | 6.6690e-04 | 1.5581e-03 | 9.9988e-03 |
| 906-bus | Penalty+SLFS | Voltage magnitude $|V|$ | 6.8597e-04 | 1.0916e-03 | 2.1279e-02 |
| 8500-node | before SLFS | Active DER $p$ | 1.2604e-04 | 3.3462e-04 | 2.7826e-03 |
| 8500-node | before SLFS | Reactive DER $q$ | 3.7225e-04 | 8.5088e-04 | 5.5705e-03 |
| 8500-node | before SLFS | Voltage magnitude $|V|$ | 7.8450e-03 | 1.0770e-02 | 5.8173e-02 |
| 8500-node | Penalty+SLFS | Active DER $p$ | 1.2928e-04 | 3.3399e-04 | 2.7820e-03 |
| 8500-node | Penalty+SLFS | Reactive DER $q$ | 3.6977e-04 | 8.4741e-04 | 5.5699e-03 |
| 8500-node | Penalty+SLFS | Voltage magnitude $|V|$ | 7.5864e-03 | 1.0640e-02 | 5.8173e-02 |
