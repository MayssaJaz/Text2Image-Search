const FetchImages = async (text, setLoading) => {
    try {
        setLoading(true);
        const response = await fetch(process.env.REACT_APP_BACK_URL + '/search/images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: text
            }),
        });
        const imagePaths = await response.json();
        // Get exact paths of images
        const imagePathsOnly = imagePaths.images.map(path => {
            return process.env.REACT_APP_IMAGES_URL + '/' + path.split('/').pop();
        });

        return imagePathsOnly;
    } catch (error) {
        console.error('Error fetching images:', error);
        return [];
    } finally {
        setLoading(false);
    }
};

export default FetchImages;