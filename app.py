import torch
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from torchtext.data.utils import get_tokenizer

# Load the trained model and vocab
batch_size = 15
lr = 1e-3
vocab_size = 743  
embedding_dim = 1024  
hidden_dim = 1024
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the vocab from vocab.pth
vocab = torch.load('vocab.pth')



# Use vocab object to extract stoi and itos
if hasattr(vocab, 'get_stoi') and hasattr(vocab, 'get_itos'):
    stoi = vocab.get_stoi()  # Get the string-to-index mapping
    itos = vocab.get_itos()  # Get the index-to-string mapping
else:
    raise AttributeError("Vocab object does not have 'get_stoi' or 'get_itos' methods!")

# Define the LSTM model architecture
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, hidden_dim).to(device),
                torch.zeros(2, batch_size, hidden_dim).to(device))

# Initialize model
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
model.to(device)

# Load the trained model
model.load_state_dict(torch.load('best-val-lstm_lm.pt', map_location=device))
model.eval()

# Tokenizer setup
tokenizer = get_tokenizer('basic_english')

# Text generation function
def generate_text(prompt, max_seq_len=1000, temperature=1.0):
    tokens = tokenizer(prompt)
    indices = [stoi.get(t, stoi['<unk>']) for t in tokens]  # Map tokens to indices
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()
            
            if prediction == stoi.get('<eos>', vocab_size - 1):  # End of sequence token
                break
            indices.append(prediction)
    
    # Convert indices back to tokens
    generated_text = " ".join([itos[i] for i in indices])
    return generated_text

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Harry Potter story Generation with LSTM"),
    dbc.Input(id="input-text", type="text", placeholder="Enter a prompt..."),
    dbc.Button("Generate", id="generate-btn", color="primary", className="mt-2"),
    html.Div(id="output-text", className="mt-3"),
])

@app.callback(
    Output("output-text", "children"),
    Input("generate-btn", "n_clicks"),
    Input("input-text", "value"),
    prevent_initial_call=True,
)
def update_output(n_clicks, prompt):
    if not prompt:
        return "Please enter a prompt."
    try:
        return generate_text(prompt)
    except Exception as e:
        return f"An error occurred while generating text: {str(e)}"

if __name__ == "__main__":
    app.run_server(debug=True)
