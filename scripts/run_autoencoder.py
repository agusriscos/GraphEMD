import torch
from GraphEMD.conf import DATA_PATH, GraphEncoderConfig, InverseFunnelDecoderConfig, TransformConfig
from GraphEMD.data import VisualGraphDataloader
from GraphEMD.model import GraphEMDAutoEncoder

if __name__ == '__main__':
    # Load autoencoder configuration
    max_num_nodes = TransformConfig.to_dict()["MAX_WINDOW_SIZE"]
    encoder_config = GraphEncoderConfig.to_dict()
    decoder_config = InverseFunnelDecoderConfig.to_dict()
    autoencoder_config = {"encoder_config": encoder_config, "decoder_config": decoder_config}
    # Load sample data
    val_dataloader = VisualGraphDataloader(
        data_dir=f"{DATA_PATH}/val_data", batch_size=1,
        shuffle=False, drop_last=False, max_num_nodes=max_num_nodes
    )
    batch = next(iter(val_dataloader))
    num_time_samples = batch.x.shape[0]
    print(f"Number of time samples (batch size): {num_time_samples}")

    model = GraphEMDAutoEncoder(autoencoder_config)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    loss_function = torch.nn.MSELoss()
    x = batch.x.reshape(-1, max_num_nodes)
    x_filter = x[x != 0]
    output_filter = output[x != 0]
    print(x_filter.size(), output_filter.size())
    loss = loss_function(output_filter, x_filter)
    print("LOSS:", loss.item())
    print("DONE!")
