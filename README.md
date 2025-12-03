# Traffic speed forecasting on the METR‑LA highway sensor network using a Residual Gated GCNN

## Abstract

*This project investigates traffic speed forecasting on the METR‑LA highway sensor network using a Residual Gated Graph CNN. The goal is to predict future traffic speeds at multiple forecasting horizons (5, 15, 30 and 60 minutes ahead) from past observations collected by 207 loop detectors in Los Angeles. The model is trained and evaluated on the public METR‑LA dataset. Experiments show that the model achieves good results, especially for short horizons, indicating that residual gated graph convolution is an effective inductive bias for traffic forecasting on road networks.*

## Problem Overview

Urban traffic forecasting is a core component of intelligent transportation systems, enabling proactive traffic management, route planning and congestion mitigation. The forecasting task is considered to be multi‑step, multi‑node regression: given a fixed‑length window of historical speeds at all sensors, the model must predict future speeds for all sensors at several horizons into the future.

Traditional time‑series models or per‑sensor deep networks struggle to exploit the underlying road network topology and spatial interactions between sensors. GNNs offer a natural way to incorporate this structure by operating on a graph whose nodes are sensors and edges encode connectivity or distance between road segments. 

This project implements and studies a Residual Gated GCNN that performs message passing over the METR‑LA sensor graph.

## What is Residual Gated GCNN?

Residual Gated GCNN is a type of GNN that improves how information flows through a graph by using gates (learnable filters controlling message flow) and residual connections (skip connections).

### Mathematical formulation

The operator implemented in PyG computes (for a node $i$):

$$ \mathbf{x}'_i = \mathbf{W}_1 \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j $$


where the gate $\eta_{i,j}$ is computed as:

$$\eta_{i,j} = \sigma(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4 \mathbf{x}_j)$$

with $\sigma$ the sigmoid function. The gates are element-wise (i.e., they have the same dimensionality as the transformed features), allowing anisotropic modulation of the neighbor messages.

**Intuition:** the gate lets the model control how much information travels from neighbor $j$ into node $i$ for each feature dimension. Residual connection (adding a transformed `x_i`) stabilizes training and helps deeper stacking.

### Key design advantages

- **Edge gating:** differentiable, learned gating that can depend on both source and target node features (and optionally edge features). Helps the network ignore irrelevant neighbors or attenuate noisy signals.
- **Residual (root) contribution:** adding a transformed version of the central (root) node features to the aggregated message — analogous to ResNet skip-connections — improves gradient flow and enables deeper GNNs.
- **Separation of key/query/value:** the implementation uses distinct linear transforms for keys, queries and values (naming inspired by attention), enabling flexible combinations of source/target representation spaces.

### When to use

- When you suspect neighbors have varying relevance and you want the model to learn per-edge importance.
- In tasks that benefit from deeper GNN stacks — residual links help with stability.

## Methodology

### Dataset

METR‑LA traffic dataset (`metr-la.h5`) containing traffic speeds (miles/hour) for 207 sensors over approximately four months (March–June 2012), sampled every 5 minutes.

Adjacency matrix (adj_mx.pkl) encodes the directed highway sensor network as:
  - `sensor_ids`: list of sensor identifiers,
  - `sensor_id_to_ind`: mapping from sensor ID to node index,
  - `adj_mx`: a 207 × 207 adjacency matrix of edge weights.

A visualization of nodes-sensors and roads between them, where the edge thickness reflects the strength/weight of the connection:

![sensors_and_roads](https://github.com/sarrtr/traffic-speed-forecasting-using-a-residual-gated-gcnn/blob/main/assets/traffic_sensor_graph.png?raw=true)

A traffic speed hratmap:

![traffic_speed_heatmap](https://github.com/sarrtr/traffic-speed-forecasting-using-a-residual-gated-gcnn/blob/main/assets/traffis_speed_heatmap.png?raw=true)
  
### Graph construction
  - From `adj_mx`, edges are extracted where the weight is positive (`adj > 0`), producing `edge_index` (pairs of node indices) and `edge_attr` (edge weights).
  - A PyTorch Geometric `Data` object is constructed with `edge_index`, `edge_attr` and `num_nodes=207`.
  - Self‑loops are included so that each sensor’s representation can depend on its own previous state.

### Model Architecture: ResidualGatedGCNN

Overall architecture is:

1. Per‑node temporal convolution over the 3‑hour history.
2. Two stacked residual gated graph convolution blocks with layer normalization and dropout.
3. Linear projection to multi‑horizon speed predictions for every sensor.

![model_architecture](https://github.com/sarrtr/traffic-speed-forecasting-using-a-residual-gated-gcnn/blob/main/assets/model_arch.png?raw=true)

### Training Process

- **Loss function**: MSE between predicted and target normalized speeds, averaged over nodes, horizons and samples.
- **Optimizer**: Adam with:
  - Learning rate ($$10^{-3}$$),
  - Weight decay ($$10^{-5}$$).
- **Learning‑rate scheduler**: `ReduceLROnPlateau` that halves the learning rate when validation loss plateaus.
- **Training length**: 10 epochs.
- **Metrics during training**:
  - After each epoch, predictions are computed on the validation set and the following metrics are calculated per horizon:
    - Mean Absolute Error (MAE),
    - Root Mean Squared Error (RMSE),
    - Coefficient of determination (R²),
    - Mean Absolute Percentage Error (MAPE, in %).

  ![train_val_loss](https://github.com/sarrtr/traffic-speed-forecasting-using-a-residual-gated-gcnn/blob/main/assets/train_val_loss.png?raw=true)

  ![train_metrics](https://github.com/sarrtr/traffic-speed-forecasting-using-a-residual-gated-gcnn/blob/main/assets/train_metrics.png?raw=true)

## Results

- **5‑minute horizon (1 step ahead)**:
  - **MAE**: 0.2021
  - **RMSE**: 0.3967
  - **R²**: 0.8433
- **15‑minute horizon (3 steps ahead)**:
  - **MAE**: 0.2409
  - **RMSE**: 0.4787
  - **R²**: 0.7719
- **30‑minute horizon (6 steps ahead)**:
  - **MAE**: 0.3051
  - **RMSE**: 0.5754
  - **R²**: 0.6703
- **60‑minute horizon (12 steps ahead)**:
  - **MAE**: 0.4097
  - **RMSE**: 0.6936
  - **R²**: 0.5210

## Discussion of Results

The experimental results demonstrate that combining temporal convolution with residual gated graph convolutions is effective for spatio‑temporal traffic forecasting on the METR‑LA network. The high R² and low error metrics at 5 and 15 minutes indicate that the model captures short‑term congestion patterns and their immediate propagation along the road network. The gradual degradation of performance for 30‑ and 60‑minute horizons is expected, reflecting the increased uncertainty and complexity of long‑range traffic dynamics, but R² values above 0.5 at one hour ahead remain practically useful in many traffic management scenarios.

There are, however, several limitations and potential improvements. First, the model uses a relatively simple temporal module (a single 1D convolution followed by temporal averaging); more expressive temporal mechanisms (e.g., temporal attention, dilated convolutions, or recurrent units) could further enhance performance, especially at longer horizons. Second, the adjacency matrix is fixed and purely spatial; dynamic or learned adjacency structures could better capture time‑varying connectivity (e.g., incidents, lane closures). Finally, the evaluation uses only standard regression metrics; downstream metrics such as congestion detection accuracy or travel‑time estimation could provide additional application‑oriented insight.

## Conclusion

This project implements and evaluates a Residual Gated Graph Convolutional Neural Network for multi‑horizon traffic speed forecasting on the METR‑LA highway sensor network. By modeling both temporal history and spatial interactions on the road graph, the approach achieves strong predictive performance, particularly for short‑term horizons, with R² above 0.84 at 5 minutes and above 0.77 at 15 minutes. The results support the hypothesis that gated message passing with residual connections is a powerful inductive bias for spatio‑temporal forecasting on transportation networks.

Beyond this specific application, the methodology illustrates a general approach for constructing graph‑based forecasting models: build a graph from domain structure, encode temporal context in node features, and apply residual gated graph convolutions to propagate information. Future work could extend this framework by incorporating more expressive temporal modules, dynamic adjacency learning and richer exogenous features (such as weather, calendar events or incident reports), as well as by benchmarking against a wider range of baselines and datasets.

## References

1. Xavier Bresson, Thomas Laurent, "Residual Gated Graph ConvNets", November 2017. 
Link: https://arxiv.org/abs/1711.07553
2. Official PyTorch Geometric implementation of 'ResGatedGraphConv`.
Link: https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/conv/res_gated_graph_conv.html#ResGatedGraphConv
3. GitHub repository of Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting with METR‑LA traffic dataset.
Link: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX