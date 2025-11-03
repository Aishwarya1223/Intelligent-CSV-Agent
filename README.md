# Intelligent CSV Analyst

An **autonomous multi-agent system** built using **Microsoft Autogen** that performs intelligent analysis, visualization, and reporting on CSV datasets — with **zero manual intervention**.  
This project demonstrates the use of **Autogen Core**, function-based tool orchestration, and cloud deployment using **AWS EC2**, **ECR**, and **S3**.

---

## 🚀 Overview

**Intelligent CSV Analyst** is a fully automated AI-powered data analysis engine that uses multiple agents to coordinate and interpret tabular data.  
Each agent has a defined role and communicates autonomously to complete an end-to-end data analysis task.

### 🧩 Agents in the System
- **Coordinator Agent** – orchestrates the overall workflow.
- **Analyst Agent** – performs data loading, cleaning, and statistical analysis.
- **Evaluator Agent** – verifies and refines results for accuracy and relevance.
- **Reporter Agent** – generates final insights and report summaries.

All agents communicate through Autogen’s internal messaging and execute analysis code dynamically using the **`autogen_core.FunctionTool`** API.

---

## ⚙️ Tools & Technologies

| Category | Tools Used |
|-----------|-------------|
| Framework | Microsoft Autogen, autogen_core |
| Programming | Python, pandas, numpy |
| AI/LLM API | OpenAI API |
| Cloud | AWS EC2, ECR, S3, Secrets Manager |
| Containerization | Docker |
| IaC | AWS CloudFormation |

---

## 🧰 Features

- ✅ Fully autonomous multi-agent system (no manual orchestration)
- ⚙️ Dynamic function execution using `autogen_core.FunctionTool`
- 📊 Automated data analysis and visualization for CSV datasets
- 🔒 Securely manages API keys via AWS Secrets Manager
- ☁️ Cloud deployment with Docker + AWS EC2 (via CloudFormation)
- 📁 Stores processed CSVs and reports in AWS S3
- 🔁 Self-restarting container (optional systemd integration)

---

## 🏗️ Architecture

                 ┌──────────────────────────────────────────────────────────────┐
                 │               AWS EC2 Instance (Docker)                      │
                 │--------------------------------------------------------------│
                 │  ┌────────────────────────────────────────────────────────┐  │
                 │  │                  Intelligent CSV Analyst               │  │
                 │  │--------------------------------------------------------│  │
                 │  │  • Autogen Agents (Coordinator, Analyst, Evaluator)    │  │
                 │  │  • FunctionTool Handlers for dynamic Python execution  │  │
                 │  │  • OpenAI API Access for reasoning and summarization   │  │
                 │  │  • CSV Analyzer for automated insights & reports       │  │
                 │  └────────────────────────────────────────────────────────┘  │
                 └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────────┐
                        │   AWS S3 Bucket (CSV Files) │
                        └─────────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────────┐
                        │ AWS Secrets Manager (API Key)│
                        └─────────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────────┐
                        │ AWS ECR Repository (Docker) │
                        └─────────────────────────────┘


---

## ☁️ Cloud Deployment Guide

### 1️⃣ Prerequisites
- AWS account with EC2, ECR, S3, and CloudFormation access  
- AWS CLI configured locally  
- Existing EC2 KeyPair  
- Docker installed on your local system  

---

### 2️⃣ Build and Push Docker Image
```bash
# Build your image
docker build -t intelligent-csv-analyst .

# Authenticate to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# Tag and push the image
docker tag intelligent-csv-analyst:latest <account-id>.dkr.ecr.<region>.amazonaws.com/intelligent-csv-analyst-repo:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/intelligent-csv-analyst-repo:latest
