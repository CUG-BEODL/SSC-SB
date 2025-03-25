import torch
import os
from argparse import ArgumentParser
from model import SimCLR
from dataloader import load_data
from model.ContrastiveLoss import ContrastiveLoss
from utils import create_directory

parser = ArgumentParser()

# Input data parameters
parser.add_argument('--len_ts', default=64, type=int, help="Length of the data")
parser.add_argument('--channels', default=10, type=int, help="Number of spectral channels in the data")
parser.add_argument('--sample_size', default=5, type=int, help="Temporal length of the short-term data")
parser.add_argument('--block_size', default=64, type=int, help="Size of a single street block")
parser.add_argument('--D_set', default=r'dataset/Changsha_developed_sampleSize_64.npy', type=str,
                    help="Path to developed samples dataset")
parser.add_argument('--UD_set', default=r'dataset/Changsha_undeveloped_sampleSize_64.npy', type=str,
                    help="Path to undeveloped samples dataset")

# Model training parameters
parser.add_argument('--device', default='cuda', type=str, help="Device for computation (e.g., 'cuda' or 'cpu')")
parser.add_argument('--proj_size', default=32, type=int, help="Projection dimension in the SimCLR model")
parser.add_argument('--batch_size', default=64, type=int, help="Batch size for training")
parser.add_argument('--max_epoch', default=200, type=int, help="Number of training epochs")
parser.add_argument('--model_save_folder', default='save_model', type=str, help="Directory to save trained models")
parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for change detection")

args = parser.parse_args()


def train(args):
    """
    Training function for SimCLR-based contrastive learning.

    Args:
        args: Parsed command-line arguments.

    Saves the model checkpoint every 5 epochs.
    """
    create_directory(args.model_save_folder)

    # Load training datasets
    train_dl = load_data(args)

    # Compute input dimension: spectral channels * temporal length
    input_dim = args.channels * args.sample_size

    # Initialize model, loss function, and optimizer
    model = SimCLR.SimCLR(input_dim=input_dim, proj_size=args.proj_size).to(args.device)
    contrastive_loss_fn = ContrastiveLoss(args.batch_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss, total_sim_pos, total_sim_neg = 0, 0, 0

        for batch_idx, (img, label) in enumerate(train_dl):
            img, label = img.to(args.device).float(), label.to(args.device).long()

            # Forward pass: obtain projection features
            _, projected_features = model(img)

            # Compute contrastive loss
            loss, sim_pos, sim_neg = contrastive_loss_fn(projected_features, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 20 batches
            if batch_idx % 20 == 0:
                print(f"> Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Accumulate statistics
            total_loss += loss.item()
            total_sim_pos += sim_pos.item()
            total_sim_neg += sim_neg.item()

        # Compute and print average loss and similarity metrics
        avg_loss = total_loss / len(train_dl)
        avg_sim_pos = total_sim_pos / len(train_dl)
        avg_sim_neg = total_sim_neg / len(train_dl)

        print(f">>>> Epoch {epoch} | Loss: {avg_loss:.4f} | Pos Sim: {avg_sim_pos:.4f} | Neg Sim: {avg_sim_neg:.4f}")

        # Save model checkpoint every 5 epochs or on the first epoch
        if epoch % 5 == 0 or epoch == 1:
            save_path = os.path.join(args.model_save_folder, f'epoch_{epoch}_{avg_loss:.4f}.pth')
            torch.save(model.state_dict(), save_path)




if __name__ == '__main__':
    train(args)  # Train the model
