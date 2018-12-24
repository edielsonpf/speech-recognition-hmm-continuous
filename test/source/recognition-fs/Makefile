CC=gcc 
CFLAGS=-pg -c -Wall
LDFLAGS= -pg -lm

SOURCES= recognition_continuous_full_fs.c

OBJECTS=$(SOURCES:.c=.o)

EXECUTABLE=../Debug/hmm-continuous-test

all: $(SOURCES) $(EXECUTABLE)
	rm -rf *.o
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(EXECUTABLE)
	rm -rf *.o
