

libpcgen.dll:
	@cl /MP /nologo /Felibpcgen.dll pcgen.c /Ox /link /MACHINE:X64 /DLL
