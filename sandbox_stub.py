"""Helper imported inside the sandbox image (see Dockerfile.sandbox).

It's copied into the image so generated code can `from stub import guard`
to confirm it's really running isolated. The executor itself enforces
`--network none --memory=512m --cpus=1`, so this is just documentation +
a convenient import that makes the isolation explicit inside generated code.
"""

GUARD = "isolated:network=none,memory=512m,cpus=1"

if __name__ == "__main__":
    print(GUARD)
