# AICareerNavigator
Use of LLM for career assisstance serve such as cover letter, application tracking, resume tailoring and etc
Current Project Structure:
aicareerchatbot/
├── app.py                 # Main Streamlit application
├── evaluation.py          # Evaluation logic
├── my_resume.pdf          # Your resume file (place it here, remove sensitive information!!!)
├── .env                   # File for your API key
└── requirements.txt       # Python dependencies

will add RAG, few shots prompting, already tried prompt engineering


Future Work
Fine tuning
now fine tuning invloves following instructions through Vertex AI(https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/rlhf_tune_llm.ipynb) or other major platforms such as OpenAI(http://platform.openai.com/docs/guides/fine-tuning), AWS also has fine tuning guide in Sagemaker(https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-fine-tune.html), the most important thing is to prepare the dataset well, otherwise, junk in junk out!!! And Fine tuning can be expensive, can try RAG, prompt engiinering. A couple of strategies for fine tuning(change learning rate and freeze front layers...)
There is also this idea of generating synthetic data from LLM, can try... but be careful about data mining other people's sensitive data in LLM!!!
