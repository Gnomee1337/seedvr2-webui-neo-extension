# <Service> — API reference

Facts about the service verified by documentation or a live request. Each fact has a source:
a link to the official documentation or "verified live" (what we sent, what we got back).
Written by the `fact-checker` agent and by the main chat after a live check. No guesses here.

## Contents
<Links to sections, if the file is long.>

## Access
What keys/tokens exist, what each can and CANNOT do, where it is stored (variable name in
`.env`, without the value), how to obtain it again.

## Common pitfalls
Things that have already burned us: an empty response instead of an error, undocumented fields, discrepancies
with the documentation.
- <Pitfall> — <how it shows up, how we work around it>. Source: <…>.

## Limits and load
- Request limit: <…> (source).
- What happens when exceeded: <response code, headers>.
- How we stay within it: <throttle in code, where>.
- Timeouts and pagination: <…>.

## <Section per entity / group of methods>
### <Method>
`<METHOD /path>` — <what it does>. Parameters: <…>. Response: <shape>. Source: <…>.

## Webhooks (if the service sends events)
Which events, body format, signature verification, retries when there is no response.

## Not verified
Open questions about the service — so they are not taken as facts.
