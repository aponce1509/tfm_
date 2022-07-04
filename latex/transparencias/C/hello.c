// #include <cs50.h>
#include <stdio.h>

int main(void)
{
    char answer[20];
    printf("What's your name?: ");
    fgets(answer, 20, stdin);
    printf("Hello, %s!\n", answer);
}