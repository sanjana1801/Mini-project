import pymongo
from pymongo import MongoClient
from pprint import pprint
client = MongoClient('localhost',27017)
db = client.food
collection = db.food

def addorder():
      cust_name=raw_input("\n\t\tEnter Customer name\t\t")
      cust_id =input("\n\t\tEnter Customer Id\t\t")
      order_no=input("\n\t\tEnter Order no\t\t")

      order_cost=input("\n\t\tEnter Order cost\t\t")
      data={
            "Customer_id":cust_id,
            "Customer_name":cust_name,
            "Order_no":order_no,
            "Order_cost":order_cost,
            "Status_of_delivery":"notDelivered"}   
      print(data)
      x=db.food.insert_one(data)
      print(x)



def showorder():
      print "\n\t\tCUSTOMERID\tCUSTOMERname\tORDERno\t\tOrdercost\tStatus"
      for s in db.food.find():         							                              
          print      "\t\t",s['Customer_id'],"\t\t",s['Customer_name'],"\t\t",s['Order_no'],"\t\t",s['Order_cost'],"\t\t",s['Status_of_delivery'],"\t\t"

def showporder():
	    w= input("Enter customer id:")      
            for s in db.food.find():
                  if s['Customer_id']== w:
               #           print s
                          print "\t\t",s['Customer_id'],"\t\t",s['Customer_name'],"\t\t",s['Order_no'],"\t\t",s['Order_cost'],"\t\t"


def updateorder():
 
	    w= input("Enter order no to be updated to delivered: ")
	    
      
      
            z=db.food.update_one({"Order_no":w},{"$set":{"Status_of_delivery":"Delivered"}})
            print(z,"updated in record")

def deleteorder():
	oid = input("Enter order id to be deleted: ")
	z=db.food.remove({"Order_no":oid})
	print(z,"order deleted")
           




def main():
      ans=0
      ch=0
      log=0
      con=1
      while(con==1):
            print("\n\t\t Enter 1 to enter MENU")
            log=int(input("\n\t\tEnter CHOICE::\t\t"))

            if(log==1):
                  while(True):
                        print("\n\t\t1.Show all Orders\n\t\t2.Add Order\n\t\t3.Update Order Status\n\t\t4.Delete Order\n\t\t5.show orders for one person")
                        ch=int(input("\n\t\tYOUR CHOICE\t::\t\t"))
                        if(ch==1):
                              showorder()
                        elif(ch==2):
                              addorder()
                        elif(ch==3):
                       	      updateorder()
                        elif(ch==4):
                              deleteorder()
                        elif(ch==5):
                              showporder()	 
                       
                        else:
                              print("\n\t\tEnter correct choice")
                        ans=int(input("\n\t\tDo u want to Continue??(1.YES)\t"))      
                        if(ans!=1):
                              break
            break  		          
                  
main()
