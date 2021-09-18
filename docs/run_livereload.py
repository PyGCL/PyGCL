from livereload import Server, shell


if __name__ == '__main__':
    server = Server()
    server.watch('*.md', shell('make html'), delay=1)
    server.watch('*.py', shell('make html'), delay=1)
    server.watch('notes/*.md', shell('make html'), delay=1)
    server.watch('modules/*.md', shell('make html'), delay=1)
    server.watch('_static/*', shell('make html'), delay=1)
    server.watch('_templates/*', shell('make html'), delay=1)
    server.serve(root='_build/html', host='*')
