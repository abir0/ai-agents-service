# 🤖 AI Agents Service

This is an AI Agents platform built using Python, FastAPI, LangChain, and Streamlit. The platform is designed to manage and deploy multiple AI agents capable of interacting with users and performing specific tasks. It supports modularity and scalability, enabling easy expansion of agents, tools, and services.

## Table of Contents
- [🤖 AI Agents Service](#-ai-agents-service)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Technologies](#technologies)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running the AI Agent Service](#running-the-ai-agent-service)
    - [Running the Streamlit Frontend](#running-the-streamlit-frontend)
  - [Docker Setup](#docker-setup)
    - [Build Docker Containers](#build-docker-containers)
    - [Start the Application](#start-the-application)
  - [Running the Project](#running-the-project)

## Project Overview

The AI Agents platform consists of several components working together:
1. **Backend (FastAPI)** - A FastAPI application that serves as the API for interacting with the AI agents.
2. **Frontend (Streamlit)** - A Streamlit web interface for users to interact with agents.
3. **Agent Service Layer** - Manages AI agents built using LangChain and LangGraph, which handle tasks like responding to user queries, performing specific actions, etc.
4. **Database Layer** - Document storage and data processing components for managing and storing the information needed by agents.

## Project Structure

```plaintext
📦 AI Agent Platform  
├── 📂 data                      # Data storage and processing  
├── 📂 logs                      # Logs directory  
├── 📂 scripts                   # Utility scripts  
├── 📂 src                       # Main application source code  
│   ├── 📂 agents                 # AI agent logic  
│   │   ├── 📂 agents_lib         # Library of agent implementations  
│   │   ├── agent_manager.py      # Manages multiple agents  
│   │   ├── client.py             # Agent client for interaction  
│   │   ├── llm.py                # LLM integration  
│   │   ├── models.py             # Data models for agents  
│   │   └── tools.py              # Custom tools for agent execution  
│   ├── 📂 api                    # FastAPI backend  
│   │   ├── 📂 endpoints          # API routes  
│   │   ├── Dockerfile            # API container setup  
│   │   ├── main.py               # API entry point  
│   │   └── utils.py              # Utility functions  
│   ├── 📂 data                   # Data management layer  
│   │   ├── 📂 loaders            # Data loading utilities  
│   │   └── 📂 processing         # Data processing logic  
│   ├── 📂 ui                     # Streamlit frontend  
│   │   ├── 📂 assets             # UI assets (images, CSS, etc.)  
│   │   ├── app.py                # Streamlit app entry point  
│   │   └── Dockerfile            # UI container setup  
│   ├── run_agent.py              # Runs an agent instance  
│   ├── run_client.py             # Starts an agent client  
│   ├── run_service.py            # Launches backend services  
│   ├── settings.py               # Configuration settings  
├── docker-compose.yaml           # Docker Compose configuration  
├── README.md                     # Project documentation  
└── requirements.txt               # Dependencies list
```

## Technologies

This project uses the following technologies:
- **Python 3.12**
- **LangGraph**
- **FastAPI**
- **Streamlit**
- **Docker**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-agent-platform.git
   cd ai-agent-platform
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the AI Agent Service

1. **Run the agent service**:
   To start the backend service, run:
   ```bash
   python src/run_service.py
   ```

2. **Run an agent**:
   You can run a specific agent by using:
   ```bash
   python src/run_agent.py
   ```

3. **Run the client**:
   To interact with an agent, start the client:
   ```bash
   python src/run_client.py
   ```

### Running the Streamlit Frontend

To start the frontend application using Streamlit, run:
```bash
streamlit run src/ui/app.py
```

## Docker Setup

### Build Docker Containers

Run the following command to build the Docker containers for both the backend and frontend:
```bash
docker compose build
```

### Start the Application

Once the containers are built, you can start the application with:
```bash
docker compose up
```

This will start both the FastAPI backend and Streamlit frontend services.

## Running the Project

To run the full stack application, use Docker Compose as described above. Alternatively, you can run the services individually by starting the backend API and Streamlit frontend separately.
