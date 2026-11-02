"""提示词模板：问答、HyDE、子查询、回溯与策略选择。"""

from idlelib import history

from langchain_core.prompts import PromptTemplate


class RAGPrompts:

    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""  
                你是一个智能助手，帮助用户回答问题。请参考用户的助手的对话历史和上下文回答问题。
                如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。  
                如果答案来源于检索到的文档，请在回答中说明。

                对话历史: {history}
                上下文: {context}  
                问题: {question}  

                如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”  
                回答:  
                """,
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            input_variables=["query"],
        )

    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            input_variables=["query"],
        )

    @staticmethod
    def backtracking_prompt():
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            input_variables=["query"],
        )

if __name__ == '__main__':
    rag_prompt = RAGPrompts.rag_prompt()
    result = rag_prompt.format(history="",context="这是一家IT培训机构,主打 Python 和 AI 课程", question="这家机构的名字叫什么?", phone="13112345678")
    print(result)
    print('♥️' * 30)
    hyde_prompt = RAGPrompts.hyde_prompt()
    result = hyde_prompt.format(query="如何培养孩子的专注力")
    print(result)
