#include<stdio.h>
#include<stdlib.h>
//Define Node
typedef struct Node{
    int data;
    struct Node*next;
}Node;
//Function to insert at the end of list
Node*insert(Node*head*,int val){
    Node*temp=(Node*)malloc(sizeof(Node));
    temp->data=val;
    temp->next=NULL;
    if(head==NULL)
    returntemp;
    Node*curr=head;
    while(curr->next!=NULL)
    curr=curr->next;
    curr->next=temp;
    return head;
}
//Function to search for a value in a listint search(Node*head,int val){
    Node*curr=head;
    while(curr!=NULL){
        if(curr->data==val){
            return 1;
            curr=curr->next;
        }
        return 0;
    }
    //a)Intersection: Both Vanilla and Butterscotch
    Node*Intersection(Node*a,Node*b){
        Node*result=NULL;
        Node*curr=a;
        while(crr!=NULL){
        if(search(b,curr->data))
        result=insert(result,curr->data);
        curr=curr->next;
        }
        return result;
    }//b)Symmetric Difference:Either but not both 
    Node*symmetric_diff(Node*a,Node*b){
        Node*result=NULL;
        Node*curr=a;
        while(curr!=NULL){
    if(!search(b,curr->data))
    result=insert(result,curr->data);
    curr=curr->next;
        }
        return result;
    }
    //Count total elements in a list
    int count(Node*head){
        int c=0;
        while(head!=NULL){
            c++;
            head=head->next;
        }
        return c;
    }
    //Display the set 
    void display(Node*head){
        while(head!=NULL){
        printf()