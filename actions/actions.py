# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

#!pip install html2text simpletransformers --quiet
# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from bs4 import BeautifulSoup
import requests
import html2text
from markdown import markdown
import re
from googlesearch import search
from google import *
from simpletransformers.question_answering import QuestionAnsweringModel


class ActionMedHelp(Action):

     def name(self) -> Text:
         return "action_med_help"
     
     def predict_answer(self,model, question, contexts, seq_len=256, debug=False):
       split_context = []
    
       if not isinstance(contexts, list):
         contexts = [contexts]
    
       for context in contexts:
         for i in range(0, len(context), seq_len):
           split_context.append(context[i:i+seq_len])
              
       split_context = contexts
       f_data = []
    
       for i, c in enumerate(split_context):
         f_data.append(
              {'qas': 
                [{'question': question,
                'id': i,
                'answers': [{'text': ' ', 'answer_start': 0}],
                'is_impossible': False}],
                'context': c
              })
        
       prediction = model.predict(f_data)
       print(type(prediction))
       if debug:
         print(prediction)
       preds = []
       preds = prediction[0][0]['answer']
    
    #preds = [preds[0].lower().strip() for x in prediction if preds[0].strip() != ' ']
       if preds:
         return sorted(list(set(preds)), key=preds.count, reverse=True)
       return 'No answer'

     def markdown_to_text(self, markdown_string):

       html = markdown(markdown_string)
       html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
       html = re.sub(r'<code>(.*?)</code >', ' ', html)

        # extract text
       soup = BeautifulSoup(html, "html.parser")
       text = ''.join(soup.findAll(text=True))

       return text

     def format_text(self, text):
       text = self.markdown_to_text(text)
       text = text.replace('\n', ' ')
       return text

     def query_pages(self, query, n=5):
       links = list(search(query, stop=n))
       links = links[::-1][0:6]
       return links

     def query_to_text(self, query, n=5):
       html_conv = html2text.HTML2Text()
       html_conv.ignore_links = True
       html_conv.escape_all = True
        
       text = []
       for link in self.query_pages(query, n):
         req = requests.get(link)
         text.append(html_conv.handle(req.text))
         text[-1] = self.format_text(text[-1])
            
       return text

     def q_to_a(self, model, question, n=5, debug=False):
       new_pred = []
       opt = ''
       context = self.query_to_text(question, n=n)
       pred = list(set(self.predict_answer(model, question, context, debug=debug)))
       for p in pred:
         new_pred.append(p.capitalize())
       opt += (p.capitalize() + '. ')
       return opt

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
             model = QuestionAnsweringModel('distilbert','distilbert-base-uncased-distilled-squad')
             recent_msg = tracker.latest_message['text']
             ans = self.q_to_a(model, recent_msg, n=8)
             dispatcher.utter_message(text=ans)
             return []
     
