# Fairness Sensitive Queries

## Dataset

The dataset consists of the annotated fairness sensitive queries, filtered from the queries of MS MARCO Passage Retrieval development set and TREC Deep Learning 2019 Passage Retrieval. The following files provide the identifier and text of the queries.

- `msmarco_passage.dev.fair.tsv`
- `trecdeep19_passage.fair.tsv`

The following files provide the full annotation of the queries, containing the corresponding categories, domains, and aspects.  

- `msmarco_passage.dev.fair.FULL_ANNOTATION.tsv`
- `trecdeep19_passage.fair.FULL_ANNOTATION.tsv`

The files has the following columns:
`QID`: identifier as provided in the original collections.
`Query`:  the free text of the query taken from the original collections.
`Categories`: Societal categories or challenges as defined in [1]. A record can contain more than one category, which in this case they are separated with comma `,`.
`Domains`: Domains as defined in the gender equality index [2].	A record can contain more than one domain, which in this case they are separated with comma `,`.
`Subdomains`: Subdomains as defined in [2] and additional facets that are commonly discussed in literature. A record can contain more than one subdomain, which in this case they are separated with comma `,`.
`Example/Description`:  A possible example or situation, describing how the biased results for this query could affect the information seeker's perception regarding societal biases, and in longer run could intensify/establish such biases in society.

## Annotation Details
Annotations you currently find in the dataset are mainly inspired by two sources: First, a review of women right and challenges provided by UN-Women [1] forms the basis for `Categories`. Second a report [2] developed by the research team of the European Institute for Gender Equality (EIGE) presents and discusses the Gender Equality Index, which we extracted the fields `Domains` and `Subdomains` from. In the table below we aim to match definitions extracted from both sources. This serves as a guideline for our annotation of socially problematic queries.

<table>
    <thead>
        <tr>
            <th>Categories</th>
            <th>Domains</th>
            <th>Subdomains</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>Career</td>
            <td rowspan=2>Work</td>
            <td>Participation</td>
        </tr>
        <tr>
            <td>Segregation and Quality of Work</td>
        </tr>
        <tr>
            <td rowspan=2>Education</td>
            <td rowspan=2>Knowledge</td>
            <td>Attainment and participation</td>
        </tr>
        <tr>
            <td>Segregation</td>
        </tr>
        <tr>
            <td rowspan=4>Social Inequality</td>
            <td rowspan=2>Time</td>
            <td>Care activities</td>
        </tr>
        <tr>
            <td>Social activities</td>
        </tr>
        <tr>
            <td rowspan=2>Money</td>
            <td>Financial Resources</td>
        </tr>
        <tr>
            <td>Economic Situation</td>
        </tr>
        <tr>
            <td rowspan=3>Politics</td>
            <td rowspan=3>Power</td>
            <td>Political</td>
        </tr>
        <tr>
            <td>Economic</td>
        </tr>
        <tr>
            <td>Social</td>
        </tr>
        <tr>
            <td rowspan=3>Health</td>
            <td rowspan=3>Health</td>
            <td>Status</td>
        </tr>
        <tr>
            <td>Behavior</td>
        </tr>
        <tr>
            <td>Access</td>
        </tr>
    </tbody>
</table>

## Disclaimer
We understand the annotation of the dataset as a continuous collaborative effort of the community. We do not argue them to be exclusive nor complete. It may broadly depend on the context of the results in which way they could influence societal norm or awareness raising of information seekers. The annotation should rather serve as an exemplary illustration of possible impacts.

## References
[1] UN Women Headquarters. 2020. UN Women Gender equality: Women’s rights in review 25 years after Beijing. https://www.unwomen.org/en/digital-library/publications/2020/03/womens-rights-in-review. Accessed: 2021-02-06.

[2] Madarova, Zuzana & Barbieri, Davide & Guidorzi, Brianna & Janeckova, Hedvika & Karu, Marre & Mollard, Blandine & Reingardė, Jolanta. (2019). Intersecting Inequalities: Gender Equality Index. 10.2839/308776.

