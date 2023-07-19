const regex = /^[a-z0-9][a-z0-9-_.\/]*$/;
const examples = [
  "abc123",
  "test-example",
  "foo_bar",
  "1abc",
  "abc-123",
  "abc.def",
  "path/to/file",
  "no spaces",
  "abc_123",
  "abc/def",
];

examples.forEach((example) => {
  console.log(`reg.test(\`${example}\`): ${regex.test(example)}`);
});
