reverse=0
a=int(input())
digit=a%10
reverse=reverse*10+digit
     if(a==reverse):
         print("yes")
     else:
         print("no")
