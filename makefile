netft: netft.o
	gcc -o netft netft.o
netft.o: netft.c
	gcc -c netft.c -o netft.o
clean:
	rm *.o *.exe
