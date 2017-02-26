
default: pcgen

pcgen:
	gcc -c -Wall -Werror -fpic -Ofast pcgen.c
	gcc -shared -o libpcgen.so pcgen.o
