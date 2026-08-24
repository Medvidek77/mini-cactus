CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -pedantic
LDFLAGS = -lm

# Check Vulkan availability
VK_EXISTS := $(shell pkg-config --exists vulkan && echo yes || echo no)
ifeq ($(VK_EXISTS),yes)
    CFLAGS += -DHAS_VULKAN $(shell pkg-config --cflags vulkan)
    LDFLAGS += $(shell pkg-config --libs vulkan)
else
    # Fallback check for system header
    VK_HDR := $(shell test -f /usr/include/vulkan/vulkan.h && echo yes || echo no)
    ifeq ($(VK_HDR),yes)
        CFLAGS += -DHAS_VULKAN
        LDFLAGS += -lvulkan
    endif
endif

# Target AVX2 if available on x86_64 host
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    CFLAGS += -mavx2 -mfma
endif

SRCS = main.c needle.c grammar.c
OBJS = $(SRCS:.c=.o)
TARGET = needle

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) needle2.bin

.PHONY: all clean
