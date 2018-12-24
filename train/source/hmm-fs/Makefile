CC=gcc 
CFLAGS=-pg -c -Wall
LDFLAGS= -pg -lm

SOURCES= ../source/hmm-fs/hmm_continuous_fs.c

OBJECTS=$(SOURCES:.c=.o)

EXECUTABLE=../Debug/hmm-continuous-train-fs

all: $(SOURCES) $(EXECUTABLE)
	rm -rf *.o
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(EXECUTABLE)
	

