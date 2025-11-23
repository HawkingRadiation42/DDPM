import re
import matplotlib.pyplot as plt

def plot_loss(log_file_path, output_image_path):
    loss_values = []
    
    # Regex to find 'loss: <number>'
    # The number can be an integer or float
    loss_pattern = re.compile(r'loss:\s*(\d+\.\d+|\d+)')
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Find all matches in the line (in case multiple updates are on one line)
                matches = loss_pattern.findall(line)
                for match in matches:
                    loss_values.append(float(match))
        
        if not loss_values:
            print("No loss values found in the file.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_image_path)
        print(f"Loss plot saved to {output_image_path}")
        print(f"Found {len(loss_values)} data points.")

    except FileNotFoundError:
        print(f"Error: File not found at {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    log_file = '/home/zjy6us/mlia/logs/mlia.err'
    output_file = '/home/zjy6us/mlia/loss_plot.png'
    plot_loss(log_file, output_file)
