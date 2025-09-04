"""
Ultra-Premium Nepal Financial Advisor - Conversational AI Chatbot
Uses LLaMA + RAG for intelligent, domain-specific financial guidance
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import time
import re
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NepalFinancialData:
    """Current Nepal financial data and domain-specific insights"""
    
    def __init__(self):
        self.load_domain_data()
        self.load_tax_rates()
        self.load_benchmarks()
    
    def load_domain_data(self):
        """Load domain-specific business data"""
        try:
            with open('data/processed/financialspecificdata.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.domain_data = {}
            self.personal_data = {}
            self.banking_data = {}
            
            for item in data:
                if 'domain' in item:
                    self.domain_data[item['domain']] = item
                elif 'personal_finance_data' in item:
                    self.personal_data = item['personal_finance_data']
                elif 'banking_data' in item:
                    self.banking_data = item['banking_data']
            
            # Add anime merchandise as clothing variant
            if 'clothing_retail' in self.domain_data:
                anime_data = self.domain_data['clothing_retail'].copy()
                anime_data['domain'] = 'anime_merchandise'
                anime_data['business_type'] = 'Anime Merchandise Store'
                # Adjust margins for niche market
                for size, data in anime_data.get('revenue_projections', {}).items():
                    if 'margin' in data:
                        data['margin'] *= 1.2  # Higher margins for specialty items
                self.domain_data['anime_merchandise'] = anime_data
            
            logger.info(f"Loaded {len(self.domain_data)} domain datasets")
        except Exception as e:
            logger.warning(f"Could not load financial data: {e}")
            self.domain_data = {}
            self.personal_data = {}
            self.banking_data = {}
    
    def load_tax_rates(self):
        """Current Nepal tax rates"""
        self.tax_rates = {
            'corporate': {'general': 0.25, 'manufacturing': 0.20, 'banks': 0.30},
            'individual': {'base': 0.01, 'max': 0.39, 'female_rebate': 0.10},
            'vat_threshold': 5000000,
            'audit_threshold': 10000000
        }
    
    def load_benchmarks(self):
        """Industry benchmarks for Nepal"""
        self.benchmarks = {
            'clothing': {'margin': 0.32, 'inventory_turns': 4, 'peak_season_boost': 2.2},
            'electronics': {'margin': 0.18, 'inventory_turns': 6, 'warranty_cost': 0.03},
            'food_delivery': {'commission': 0.18, 'delivery_cost': 0.08, 'customer_retention': 0.65},
            'fintech': {'transaction_fee': 0.025, 'compliance_cost': 0.15, 'user_acquisition': 200},
            'consultancy': {'utilization_rate': 0.75, 'hourly_billing': 2500, 'repeat_clients': 0.40}
        }

class ConversationalAdvisor:
    """Intelligent conversational financial advisor using LLaMA"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nepal_data = NepalFinancialData()
        self.conversation_context = {}
        self._load_models()
        self._load_rag_data()
    
    def _load_models(self):
        """Load LLaMA model for conversation"""
        logger.info("Loading LLaMA model...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info("LLaMA model loaded successfully")
    
    def _load_rag_data(self):
        """Load RAG documents for legal/tax context"""
        try:
            with open('data/processed/document_chunks.json', 'r', encoding='utf-8') as f:
                self.legal_chunks = json.load(f)
            logger.info(f"Loaded {len(self.legal_chunks)} legal document chunks")
        except Exception as e:
            logger.warning(f"Could not load legal documents: {e}")
            self.legal_chunks = []
    
    def extract_user_context(self, message: str) -> Dict[str, Any]:
        """Extract business and personal context from user message using LLaMA"""
        
        extraction_prompt = f"""Extract key information from this user message about their business or personal finances:

User Message: "{message}"

Extract and categorize the following information (use "NA" if not mentioned):

BUSINESS INFO:
- Domain: (clothing/electronics/food_delivery/fintech/consultancy/other)
- Stage: (planning/startup/existing/scaling)
- Monthly Revenue: (amount in NPR or "NA")
- Location: (city/area or "NA")
- Employees: (number or "NA")
- Capital Available: (amount or "NA")

PERSONAL INFO:
- Age: (range like 25-30 or "NA")
- Marital Status: (single/married or "NA")
- Monthly Income: (amount or "NA")
- Goals: (house/car/investment/education or "NA")
- Risk Appetite: (low/medium/high or "NA")

OUTPUT FORMAT:
Business: domain=X, stage=X, revenue=X, location=X, employees=X, capital=X
Personal: age=X, status=X, income=X, goals=X, risk=X

Extract concisely:"""

        inputs = self.tokenizer(extraction_prompt, return_tensors="pt", truncation=True, max_length=1500)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,  # Reduced for faster extraction
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse the structured response
        context = self._parse_extracted_context(response)
        self.conversation_context.update(context)
        
        return context
    
    def _parse_extracted_context(self, llama_response: str) -> Dict[str, Any]:
        """Parse LLaMA's structured extraction"""
        context = {
            'business': {},
            'personal': {},
            'needs_info': []
        }
        
        try:
            lines = llama_response.strip().split('\n')
            for line in lines:
                if 'Business:' in line:
                    # Parse business info
                    parts = line.split('Business:')[1].split(',')
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if value != "NA" and value != "X":
                                context['business'][key] = value
                            else:
                                context['needs_info'].append(f"business_{key}")
                
                elif 'Personal:' in line:
                    # Parse personal info
                    parts = line.split('Personal:')[1].split(',')
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if value != "NA" and value != "X":
                                context['personal'][key] = value
                            else:
                                context['needs_info'].append(f"personal_{key}")
        
        except Exception as e:
            logger.warning(f"Error parsing context: {e}")
        
        return context
    
    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """Get specific insights for business domain"""
        if domain in self.nepal_data.domain_data:
            domain_info = self.nepal_data.domain_data[domain]
            benchmarks = self.nepal_data.benchmarks.get(domain, {})
            
            return {
                'startup_costs': domain_info.get('startup_costs', {}),
                'revenue_projections': domain_info.get('revenue_projections', {}),
                'location_data': domain_info.get('location_data', {}),
                'benchmarks': benchmarks,
                'suppliers': domain_info.get('supplier_info', {}),
                'staff_costs': domain_info.get('staff_costs', {}),
                'seasonal_factors': domain_info.get('seasonal_factors', {})
            }
        return {}
    
    def calculate_projections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial projections based on context"""
        projections = {}
        
        business = context.get('business', {})
        domain = business.get('domain', 'general')
        
        # Get domain-specific data
        domain_insights = self.get_domain_insights(domain)
        
        if domain_insights:
            startup_costs = domain_insights.get('startup_costs', {})
            revenue_data = domain_insights.get('revenue_projections', {})
            
            # Calculate startup investment
            if startup_costs:
                total_min = sum(cost.get('min', 0) for cost in startup_costs.values() if isinstance(cost, dict))
                total_max = sum(cost.get('max', 0) for cost in startup_costs.values() if isinstance(cost, dict))
                projections['startup_investment'] = {'min': total_min, 'max': total_max}
            
            # Revenue projections
            if revenue_data:
                projections['revenue_scenarios'] = revenue_data
            
            # Loan eligibility
            if business.get('revenue'):
                try:
                    monthly_revenue = float(re.sub(r'[^\d.]', '', str(business['revenue'])))
                    loan_eligibility = monthly_revenue * 6  # Conservative 6x monthly revenue
                    projections['loan_eligibility'] = loan_eligibility
                except:
                    pass
        
        return projections
    
    def generate_advice(self, user_message: str, context: Dict[str, Any], projections: Dict[str, Any]) -> str:
        """Generate comprehensive advice using LLaMA"""
        
        # Prepare context for LLaMA
        business_context = context.get('business', {})
        personal_context = context.get('personal', {})
        domain = business_context.get('domain', 'general')
        
        # Get domain insights
        domain_insights = self.get_domain_insights(domain)
        
        # Get relevant legal chunks
        legal_context = self._get_relevant_legal_context(user_message)
        
        # Ultra-premium system prompt
        system_prompt = f"""You are an ultra-premium personal and business financial advisor for clients in Nepal. 
Provide precise, actionable, realistic guidance based on local data.

CURRENT CLIENT CONTEXT:
Business: {business_context}
Personal: {personal_context}
Domain Insights: {domain_insights}
Projections: {projections}

NEPAL FINANCIAL DATA:
- Corporate tax: 25% general, 20% manufacturing/IT, 30% banks
- VAT threshold: NPR 50 lakhs annual turnover
- Business loan rates: 9-13% (secured), 12-16% (unsecured)
- Individual tax: Progressive 1%-39%, female rebate 10%

LEGAL CONTEXT (if relevant):
{legal_context[:500] if legal_context else "None"}

USER QUESTION: {user_message}

Provide guidance in these sections:
1. BUSINESS GUIDANCE (if applicable)
2. PERSONAL FINANCIAL GUIDANCE (if applicable)  
3. ACTIONABLE NEXT STEPS (3-5 concrete steps)

Make your response:
- Specific to Nepal market conditions
- Include actual numbers and ranges
- Provide realistic timelines
- Address their specific domain ({domain})
- Be conversational but professional
- Ask for missing critical information if needed

RESPONSE:"""

        inputs = self.tokenizer(system_prompt, return_tensors="pt", truncation=True, max_length=2000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Reduced for faster responses
                temperature=0.2,     # Lower for faster, focused responses
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _get_relevant_legal_context(self, query: str) -> str:
        """Get relevant legal documents for context"""
        if not self.legal_chunks:
            return ""
        
        query_lower = query.lower()
        relevant_chunks = []
        
        # Simple keyword matching
        legal_keywords = ['tax', 'registration', 'license', 'compliance', 'penalty', 'law', 'rule']
        
        if any(keyword in query_lower for keyword in legal_keywords):
            for chunk in self.legal_chunks[:5]:  # Top 5 chunks
                if any(keyword in chunk['text'].lower() for keyword in legal_keywords):
                    relevant_chunks.append(chunk['text'][:200])
        
        return "\n".join(relevant_chunks)
    
    def chat(self, user_message: str) -> str:
        """Main chat interface"""
        logger.info(f"Processing: {user_message}")
        
        # Extract context from user message
        context = self.extract_user_context(user_message)
        
        # Calculate projections
        projections = self.calculate_projections(context)
        
        # Generate advice
        advice = self.generate_advice(user_message, context, projections)
        
        return advice

def main():
    """Interactive chatbot interface"""
    print("🇳🇵 Ultra-Premium Nepal Financial Advisor")
    print("=" * 50)
    print("Your personal AI financial advisor for business and personal finances in Nepal")
    print("Ask me anything about starting a business, investments, taxes, loans, or financial planning!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    try:
        advisor = ConversationalAdvisor()
        print("✅ Ready! I'm here to help with your financial questions.")
        
        # Welcome message
        print("\n🤝 Hi! I'm your financial advisor. To give you the best advice, I'd love to know:")
        print("- What type of business are you interested in? (clothing, electronics, food delivery, etc.)")
        print("- Are you planning to start, already running, or looking to scale?")
        print("- What's your current monthly income or revenue?")
        print("- What are your main financial goals?")
        print("\nBut feel free to ask me anything directly - I'll work with whatever information you provide!")
        
        conversation_count = 0
        
        while True:
            user_input = input(f"\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using Nepal Financial Advisor! Best of luck with your financial journey!")
                break
            
            if not user_input:
                continue
            
            conversation_count += 1
            print(f"\n🤔 Thinking... (Conversation #{conversation_count})")
            
            try:
                start_time = time.time()
                response = advisor.chat(user_input)
                processing_time = time.time() - start_time
                
                print(f"\n💡 Advisor:")
                print(response)
                print(f"\n⏱️ Response time: {processing_time:.1f}s")
                
                # Suggest follow-up questions
                if conversation_count == 1:
                    print(f"\n💭 Follow-up suggestions:")
                    print("- 'What loan options do I have?'")
                    print("- 'Help me create a budget plan'")
                    print("- 'What are the tax implications?'")
                    print("- 'How should I invest my savings?'")
                
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print(f"❌ Sorry, I encountered an error: {e}")
                print("Please try rephrasing your question.")
    
    except Exception as e:
        logger.error(f"Failed to initialize advisor: {e}")
        print(f"❌ Failed to start the advisor: {e}")
        print("Please check if all required files are available.")

if __name__ == "__main__":
    main()
