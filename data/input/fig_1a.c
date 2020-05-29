#include <stdio.h>

void main (){
	int i, n=5, s=0;
	for (i=1, i<n, i++)
		s = s+i*(i++) /2;
		
	printf ("%d", s);
}