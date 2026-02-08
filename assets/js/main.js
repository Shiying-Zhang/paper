document.addEventListener('DOMContentLoaded', function () {
  const searchInput = document.getElementById('searchInput');
  const tagButtons = document.querySelectorAll('.tag-btn');
  const sortSelect = document.getElementById('sortSelect');
  const tableBody = document.querySelector('#paperTable tbody');
  const noResults = document.getElementById('noResults');

  if (!tableBody) return;

  let activeTag = 'all';

  // Search
  if (searchInput) {
    searchInput.addEventListener('input', applyFilters);
  }

  // Tag filter
  tagButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      tagButtons.forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      activeTag = btn.getAttribute('data-tag');
      applyFilters();
    });
  });

  // Sort
  if (sortSelect) {
    sortSelect.addEventListener('change', applyFilters);
  }

  function applyFilters() {
    var keyword = searchInput ? searchInput.value.toLowerCase().trim() : '';
    var rows = Array.from(tableBody.querySelectorAll('.paper-row'));
    var visibleCount = 0;

    rows.forEach(function (row) {
      var title = row.getAttribute('data-title') || '';
      var authors = row.getAttribute('data-authors') || '';
      var tags = row.getAttribute('data-tags') || '';
      var venue = (row.getAttribute('data-venue') || '').toLowerCase();

      // Keyword filter
      var matchKeyword = !keyword ||
        title.indexOf(keyword) !== -1 ||
        authors.indexOf(keyword) !== -1 ||
        tags.indexOf(keyword) !== -1 ||
        venue.indexOf(keyword) !== -1;

      // Tag filter
      var matchTag = activeTag === 'all' || tags.indexOf(activeTag.toLowerCase()) !== -1;

      if (matchKeyword && matchTag) {
        row.style.display = '';
        visibleCount++;
      } else {
        row.style.display = 'none';
      }
    });

    // Sort visible rows
    var sortValue = sortSelect ? sortSelect.value : 'year-desc';
    var sortedRows = rows.slice().sort(function (a, b) {
      switch (sortValue) {
        case 'year-desc':
          return parseInt(b.getAttribute('data-year')) - parseInt(a.getAttribute('data-year'));
        case 'year-asc':
          return parseInt(a.getAttribute('data-year')) - parseInt(b.getAttribute('data-year'));
        case 'rating-desc':
          return parseInt(b.getAttribute('data-rating')) - parseInt(a.getAttribute('data-rating'));
        case 'rating-asc':
          return parseInt(a.getAttribute('data-rating')) - parseInt(b.getAttribute('data-rating'));
        case 'title-asc':
          return (a.getAttribute('data-title') || '').localeCompare(b.getAttribute('data-title') || '');
        default:
          return 0;
      }
    });

    sortedRows.forEach(function (row) {
      tableBody.appendChild(row);
    });

    // Show/hide no results
    if (noResults) {
      noResults.style.display = visibleCount === 0 ? 'block' : 'none';
    }
  }
});
