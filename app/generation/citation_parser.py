import re


class CitationParser:
    """
    Extracts individual claims together with
    their citations.

    Example:

    Employees receive 24 leaves [1].
    Remote work is allowed [2][3].

    ->
    [
        ("Employees receive 24 leaves", [1]),
        ("Remote work is allowed", [2,3])
    ]
    """

    citation_pattern = re.compile(r"\[(\d+)\]")

    def parse(
        self,
        answer: str,
    ) -> list[tuple[str, list[int]]]:

        claims = []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer.strip(),
        )

        for sentence in sentences:

            citations = [

                int(c)

                for c in self.citation_pattern.findall(
                    sentence
                )

            ]

            clean_sentence = self.citation_pattern.sub(
                "",
                sentence,
            ).strip()

            if clean_sentence:

                claims.append(
                    (
                        clean_sentence,
                        citations,
                    )
                )

        return claims