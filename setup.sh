
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if ! command_exists conda; then
    echo "conda could not be found. Please install Anaconda or Miniconda first."
    echo "You can download Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

if ! command_exists wget; then
    echo "wget could not be found. Please install it using one of the following commands:"
    echo "For Ubuntu/Debian:"
    echo "  sudo apt-get update && sudo apt-get install wget"
    echo "For Fedora:"
    echo "  sudo dnf install wget"
    echo "For CentOS/RHEL:"
    echo "  sudo yum install wget"
    echo "For Arch Linux:"
    echo "  sudo pacman -S wget"
    echo "For macOS (using Homebrew):"
    echo "  First, install Homebrew from https://brew.sh/"
    echo "  Then run: brew install wget"
    exit 1
fi

echo "Downloading MovieLens dataset..."
if wget https://files.grouplens.org/datasets/movielens/ml-latest.zip; then
    echo "Download completed successfully."
else
    echo "Failed to download the dataset. Please check your internet connection and try again."
    exit 1
fi

echo "Extracting the dataset..."
if unzip ml-latest.zip; then
    echo "Extraction completed successfully."
else
    echo "Failed to extract the dataset. Please make sure unzip is installed and try again."
    exit 1
fi

echo "Creating conda environment..."
if conda create -n movie_recommender python=3.8 -y; then
    echo "Conda environment 'movie_recommender' created successfully."
else
    echo "Failed to create conda environment. Please check the error messages above and try again."
    exit 1
fi

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate movie_recommender

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Please check the error messages above and try again."
    exit 1
fi

echo "Conda environment 'movie_recommender' activated."

echo "Creating .env file..."
touch .env
echo ".env file created. Please edit it with your database and API credentials."

echo "Installing Python requirements..."
pip freeze > requirements.txt
if pip install -r requirements.txt; then
    echo "Python requirements installed successfully."
else
    echo "Failed to install Python requirements. Please check the error messages above and try again."
    exit 1
fi

echo "Running data preprocessing and insertion script..."
if python preprocess_and_insert_to_DB.py; then
    echo "Data preprocessing and insertion completed successfully."
else
    echo "Failed to preprocess and insert data. Please check the error messages above and try again."
    exit 1
fi

echo "Starting FastAPI service..."
python app/fastapi_endpoint_2.py &
FASTAPI_PID=$!
echo "FastAPI service started with PID $FASTAPI_PID"

echo "Starting Streamlit app..."
if streamlit run app/streamlit_app_v3.py; then
    echo "Streamlit app started successfully."
else
    echo "Failed to start Streamlit app. Please check the error messages above and try again."
    kill $FASTAPI_PID
    exit 1
fi

echo "Setup completed successfully. You can now access the FastAPI service and Streamlit app."
