"""Composer-to-era mapping for the hierarchical evaluation."""

COMPOSER_ERA = {
    # Baroque
    'Bach':         'Baroque',
    'Handel':       'Baroque',
    'Scarlatti':    'Baroque',
    'Couperin':     'Baroque',
    'Rameau':       'Baroque',
    'Telemann':     'Baroque',
    'Buxtehude':    'Baroque',

    # Classical
    'Mozart':       'Classical',
    'Haydn':        'Classical',
    'Clementi':     'Classical',
    'Beethoven':    'Classical',   # debatable — early Beethoven is Classical

    # Romantic
    'Chopin':       'Romantic',
    'Schumann':     'Romantic',
    'Liszt':        'Romantic',
    'Brahms':       'Romantic',
    'Mendelssohn':  'Romantic',
    'Schubert':     'Romantic',
    'Tchaikovsky':  'Romantic',
    'Grieg':        'Romantic',
    'Rachmaninoff': 'Romantic',
    'Rachmaninov':  'Romantic',
    'Faure':        'Romantic',
    'Fauré':        'Romantic',
    'Albeniz':      'Romantic',
    'Albéniz':      'Romantic',

    # Impressionist / Early modern
    'Debussy':      'Impressionist',
    'Ravel':        'Impressionist',
    'Satie':        'Impressionist',
    'Scriabin':     'Impressionist',
    'Granados':     'Impressionist',

    # 20th century / Modern
    'Prokofiev':    'Modern',
    'Bartok':       'Modern',
    'Bartók':       'Modern',
    'Shostakovich': 'Modern',
    'Stravinsky':   'Modern',
    'Ligeti':       'Modern',
    'Messiaen':     'Modern',
    'Cage':         'Modern',
    'Glass':        'Modern',
    'Hindemith':    'Modern',
}


def era_for(composer: str) -> str:
    """Return era for a composer surname, or 'Unknown' if not in the lookup."""
    if not isinstance(composer, str):
        return 'Unknown'
    for key, era in COMPOSER_ERA.items():
        if key.lower() in composer.lower():
            return era
    return 'Unknown'
