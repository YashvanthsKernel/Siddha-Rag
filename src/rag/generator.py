"""
Response Generator Module
Generates responses using Ollama LLM with retrieved context.
"""

import ollama
from typing import List, Dict, Optional


class ResponseGenerator:
    """
    Generates responses using Ollama LLM models.
    Implements RAG (Retrieval-Augmented Generation) pattern.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the response generator.
        
        Args:
            model_name: Name of Ollama model to use
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            system_prompt: Optional system prompt to set model behavior
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Default system prompt for Siddha medicine expert
        self.system_prompt = system_prompt or """You are an expert in Siddha medicine, an ancient Indian system of medicine. 
Your role is to provide accurate, helpful information based on the context provided from authentic Siddha medical texts.

Guidelines:
1. Answer based primarily on the provided context
2. If the context doesn't contain relevant information, say so honestly
3. Use clear, accessible language while maintaining medical accuracy
4. Cite sources when available in the context
5. If asked about medical treatment, remind users to consult qualified practitioners"""
        
        print(f"✅ Initialized ResponseGenerator with model: {model_name}")
        print(f"   Temperature: {temperature}")
    
    def generate(
        self,
        query: str,
        context_chunks: List[str],
        include_sources: bool = True,
        max_context_length: int = 4000
    ) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User's question
            context_chunks: List of retrieved document chunks
            include_sources: Whether to ask LLM to cite sources
            max_context_length: Maximum characters to use from context
            
        Returns:
            Generated response
        """
        # Build context string
        context = "\n\n".join(context_chunks)
        
        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n[Context truncated...]"
        
        # Build prompt
        citations_instruction = "\nCite the sources when providing information." if include_sources else ""
        
        prompt = f"""Based on the following context from Siddha medical texts, please answer the question.

Context:
{context}

Question: {query}
{citations_instruction}

Answer:"""
        
        print(f"\n🤖 Generating response with {self.model_name}...")
        
        try:
            # Query Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self.system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': self.temperature
                }
            )
            
            answer = response['message']['content']
            print(f"✅ Response generated ({len(answer)} characters)")
            
            return answer
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {e}"
            print(error_msg)
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def generate_with_chat_history(
        self,
        query: str,
        context_chunks: List[str],
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate response with conversation history.
        
        Args:
            query: Current user question
            context_chunks: Retrieved context
            chat_history: Previous messages [{'role': 'user'/'assistant', 'content': '...'}]
            
        Returns:
            Generated response
        """
        # Build context
        context = "\n\n".join(context_chunks)
        
        # Build messages
        messages = [{'role': 'system', 'content': self.system_prompt}]
        
        # Add chat history
        if chat_history:
            messages.extend(chat_history)
        
        # Add current query with context
        current_prompt = f"""Context from Siddha texts:
{context}

Question: {query}"""
        
        messages.append({
            'role': 'user',
            'content': current_prompt
        })
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={'temperature': self.temperature}
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_simple(self, query: str, context: str) -> str:
        """
        Simple generation without RAG setup (for quick testing).
        
        Args:
            query: User question
            context: Context string
            
        Returns:
            Generated response
        """
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def stream_generate(
        self,
        query: str,
        context_chunks: List[str]
    ):
        """
        Generate response with streaming (yields chunks as they're generated).
        
        Args:
            query: User question
            context_chunks: Retrieved context
            
        Yields:
            Response chunks as they're generated
        """
        context = "\n\n".join(context_chunks)
        
        prompt = f"""Context: {context}

Question: {query}

Answer:"""
        
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            yield f"Error: {str(e)}"


def main():
    """
    Example usage and testing
    """
    # Sample context chunks (would come from retriever in real usage)
    sample_context = [
        "Neem (Azadirachta indica) is extensively used in Siddha medicine for treating various skin diseases including eczema, psoriasis, and acne. The leaves have antibacterial and antifungal properties.",
        "In Siddha texts, Neem is classified as a blood purifier. It is recommended for chronic skin conditions and is often combined with turmeric for enhanced effectiveness.",
        "The bark of Neem tree is used to prepare decoctions for treating fever and malaria in traditional Siddha practice."
    ]
    
    # Initialize generator
    generator = ResponseGenerator(model_name="llama3", temperature=0.7)
    
    # Test query
    query = "What are the medicinal uses of Neem in Siddha medicine?"
    
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)
    
    # Generate response
    response = generator.generate(query, sample_context)
    
    print(f"\n📝 Response:\n{response}")
    
    # Test streaming
    print("\n" + "="*80)
    print("Testing streaming response:")
    print("="*80 + "\n")
    
    for chunk in generator.stream_generate(query, sample_context):
        print(chunk, end='', flush=True)
    
    print("\n\n✅ Generator test complete!")


if __name__ == "__main__":
    main()
