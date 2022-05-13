
import firebase_admin
from firebase import firebase
from firebase_admin import credentials
from firebase_admin import firestore

firebase = firebase.FirebaseApplication("https://myhand-ff333-default-rtdb.firebaseio.com/")

anelar      = firebase.put('/mao', 'anelar', 0)
indicador   = firebase.put('/mao', 'indicador', 0)
meio        = firebase.put('/mao', 'meio', 0)
mindinho    = firebase.put('/mao', 'mindinho', 0)
polegar     = firebase.put('/mao', 'polegar', 0)

print(firebase.get("https://myhand-ff333-default-rtdb.firebaseio.com/", "/mao"))

""" 
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

db.collection('hand').add({'thumb finger':1,
                            'index finger':1,
                            'middle finger':1,
                            'ringe finger':1,
                            'little finger':1}) """