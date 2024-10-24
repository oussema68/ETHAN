# 📄 Flask-Based AI PDF Processing App

This repository contains a Flask-based web application for uploading, processing, and interacting with PDF files using GPT-2 embeddings and a custom question-answering flow. It also supports user management with authentication, file uploads to AWS S3, and chat data persistence with DynamoDB.

## 🌟 Features

- **User Authentication**: Sign-up, login, and password reset functionalities with secure password hashing.
- **PDF Uploads**: Users can upload PDF files which are stored in AWS S3.
- **AI-Driven Q&A**: The app leverages GPT-2 embeddings to answer questions from the uploaded PDFs.
- **Session Management**: User sessions are tracked, and responses are stored securely in DynamoDB.
- **Recent Chats**: Users can view and download their recent chat history.
  
## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed and configured:

- **Python 3.8+**
- **Flask** for web development
- **PyPDF2** for PDF extraction
- **AWS SDK (boto3)** for AWS S3 and DynamoDB
- **Transformers (HuggingFace)** for GPT-2 model integration
- **LangChain** for chain-based processing
- **FAISS** for fast vector similarity search
- **AWS S3** bucket and **DynamoDB** tables for storage

### 🛠 Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies**:

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:

   Create a `.env` file in the root directory and set up your environment variables:

   ```bash
   OPENAI_API_KEY=<your-openai-api-key>
   AWS_ACCESS_KEY=<your-aws-access-key>
   AWS_SECRET_KEY=<your-aws-secret-key>
   REGION_NAME=<your-aws-region>
   BUCKET_NAME=<your-s3-bucket-name>
   FLASK_SECRET_KEY=<your-flask-secret-key>
   ```

4. **DynamoDB Tables**:

   Ensure that your AWS DynamoDB tables are set up for storing user accounts and responses:

   - **UserAccounts**: For managing user details.
   - **UserResponses**: For storing user-specific chat responses.

### 💻 Running the App

To run the application locally:

```bash
python app.py
```

The app will start at `http://localhost:5001`.

### 📂 File Upload and Processing

1. **Upload PDFs**: Users can upload PDF files via the app's interface. These files are uploaded to your configured AWS S3 bucket.
2. **AI Questioning**: Once a PDF is uploaded, users can ask questions based on the content, and the app will use GPT-2 embeddings to return answers.

### 🛡️ User Authentication

- **Sign Up**: Users can create accounts with username and password.
- **Login**: Secure authentication is provided using hashed passwords.
- **Password Reset**: Users can reset their passwords through the app interface.

### 🧠 AI-Powered PDF Questioning

The AI module uses **GPT-2** for embedding the text extracted from PDFs and a **k-nearest neighbors (KNN)** algorithm to perform similarity searches. The app can answer user queries by processing the text chunks of uploaded PDFs.

### 📊 Data Storage

- **S3**: Stores the uploaded PDFs.
- **DynamoDB**: Stores user information and recent chat history.

### 📝 Recent Chats

- Users can view their most recent 20 chat sessions directly from the app.
- A downloadable text file of recent chats is available.

## ⚙️ Key Functions

- **User Management**: `login_user`, `signup_user`, `reset_password`, `logout`
- **PDF Handling**: `upload_file_to_s3`, `download_file_from_s3`, `get_gpt2_embeddings`
- **Chat Persistence**: `get_user_responses`, `set_user_responses`
- **AI Integration**: `VectorStore` for vector similarity search, GPT-2 model integration.

## 🤖 AI Model Usage

This app uses the Hugging Face GPT-2 model for text embedding and question-answering processes.

```python
# Example function for getting embeddings
def get_gpt2_embeddings(text):
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = tokenized_text["input_ids"]
    with torch.no_grad():
        embeddings = model(input_ids).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()
```

## 🖼️ Screenshots

### Login / Signup Page
A secure login and registration page with session management.

### PDF Upload and Questioning
Upload a PDF, ask questions, and receive AI-generated answers.

### Recent Chats
View and download a log of recent conversations.

## 🌐 Deployment

You can deploy this app using any WSGI-compatible platform such as:

- **Heroku**
- **AWS Elastic Beanstalk**
- **Google Cloud Platform**

Ensure your environment variables are configured properly in the respective platform.

## 📝 License

This project is licensed under the MIT License.

## ✨ Contributions

Feel free to fork the repository, submit issues, or make pull requests!

---

Made with ❤️ by [Your Name].
