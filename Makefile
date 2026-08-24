CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -pedantic
LDFLAGS = -lm

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
CFLAGS += -mavx2 -mfma
endif

VK_EXISTS := $(shell pkg-config --exists vulkan 2>/dev/null && echo yes || echo no)
ifeq ($(VK_EXISTS),yes)
CFLAGS += -DHAS_VULKAN $(shell pkg-config --cflags vulkan)
LDFLAGS += $(shell pkg-config --libs vulkan)
else
VK_HDR := $(shell test -f /usr/include/vulkan/vulkan.h -o -f /usr/local/include/vulkan/vulkan.h && echo yes || echo no)
ifeq ($(VK_HDR),yes)
CFLAGS += -DHAS_VULKAN
LDFLAGS += -lvulkan
endif
endif

SRCS = main.c needle.c grammar.c
OBJS = main.o needle.o grammar.o
TARGET = needle

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

main.o: main.c needle.h
	$(CC) $(CFLAGS) -c main.c -o main.o

needle.o: needle.c needle.h grammar.h
	$(CC) $(CFLAGS) -c needle.c -o needle.o

grammar.o: grammar.c grammar.h
	$(CC) $(CFLAGS) -c grammar.c -o grammar.o

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
